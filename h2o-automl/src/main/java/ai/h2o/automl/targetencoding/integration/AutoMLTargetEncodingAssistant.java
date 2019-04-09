package ai.h2o.automl.targetencoding.integration;

import ai.h2o.automl.targetencoding.BlendingParams;
import ai.h2o.automl.targetencoding.TargetEncoder;
import ai.h2o.automl.targetencoding.TargetEncodingParams;
import ai.h2o.automl.targetencoding.strategy.*;
import hex.ModelBuilder;
import hex.splitframe.ShuffleSplitFrame;
import water.DKV;
import water.Key;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.Log;
import water.util.StringUtils;
import water.util.TwoDimTable;

import java.util.HashMap;
import java.util.Map;

import static ai.h2o.automl.targetencoding.TargetEncoder.DataLeakageHandlingStrategy.KFold;
import static ai.h2o.automl.targetencoding.TargetEncoderFrameHelper.addKFoldColumn;
import static ai.h2o.automl.targetencoding.TargetEncoderFrameHelper.concat;

/**
 * Dedicated to a single ModelBuilder, i.e. model with hyperParameters
 * Perform TargetEncoding based on a strategy of searching best TE parameters. 
 * Side effects will be done mostly through mutating modelBuilder.
 */
public class AutoMLTargetEncodingAssistant{

  private Frame _trainingFrame;   
  private Frame _validationFrame; 
  private Frame _leaderboardFrame;
  private String _responseColumnName;
  private AutoMLBuildSpec _buildSpec;
  private ModelBuilder _modelBuilder;
  private boolean _CVEarlyStoppingEnabled;

  public TEParamsSelectionStrategy getTeParamsSelectionStrategy() {
    return _teParamsSelectionStrategy;
  }

  private TEParamsSelectionStrategy _teParamsSelectionStrategy;
  
  private TEApplicationStrategy _applicationStrategy;
  private String[] _columnsToEncode;

  public TargetEncodingParams getBestTEParams() {
    return _teParams;
  }

  // This field will be initialised with the optimal target encoding params returned from TEParamsSelectionStrategy
  private TargetEncodingParams _teParams;

  public AutoMLTargetEncodingAssistant(Frame trainingFrame, // maybe we don't need all these as we are working with particular modelBuilder and not the main AutoML data
                                Frame validationFrame,
                                Frame leaderboardFrame,
                                AutoMLBuildSpec buildSpec,
                                ModelBuilder modelBuilder) {
    _modelBuilder = modelBuilder;
    _trainingFrame = modelBuilder.train();
    _validationFrame = validationFrame;
    _leaderboardFrame = leaderboardFrame;
    _responseColumnName = _modelBuilder._parms._response_column;

    _buildSpec = buildSpec;

    _CVEarlyStoppingEnabled = _modelBuilder._parms.valid() == null;

  }

  public void init() {
    // Application strategy
    TEApplicationStrategy applicationStrategy = _buildSpec.te_spec.application_strategy;
    _applicationStrategy = applicationStrategy != null ? applicationStrategy : new AllCategoricalTEApplicationStrategy(_trainingFrame, _responseColumnName);
    _columnsToEncode = _applicationStrategy.getColumnsToEncode();

    //TODO what is the canonical way to get metric we are going to use. DistributionFamily, leaderboard metrics?
    boolean theBiggerTheBetter = _modelBuilder._parms.train().vec(_responseColumnName).get_type() != Vec.T_NUM;
    double ratioOfHyperspaceToExplore = _buildSpec.te_spec.ratio_of_hyperspace_to_explore;

    // Selection strategy
    HPsSelectionStrategy teParamsSelectionStrategy = _buildSpec.te_spec.params_selection_strategy;
    switch(teParamsSelectionStrategy) {
      case Fixed:
        TargetEncodingParams targetEncodingParams = new TargetEncodingParams(_columnsToEncode, new BlendingParams(5, 1), TargetEncoder.DataLeakageHandlingStrategy.KFold, 0.01);
        _teParamsSelectionStrategy = new FixedTEParamsStrategy(targetEncodingParams);
        break;
      case RGS:
      default:
        //After filtering out some categorical columns with `applicationStrategy` we can try to search for optimal combinationof the rest as well. 
        // This covers the case with no columns to encode, i.e. no target encoding
        Map<String, Double> _columnNameToIdxMap = new HashMap<>();//leaderboard.find(_columnsToEncode);
        for (String column : _columnsToEncode) {
          _columnNameToIdxMap.put(column, (double) _trainingFrame.find(column));
        }
        _teParamsSelectionStrategy = new GridSearchTEParamsSelectionStrategy(_leaderboardFrame, ratioOfHyperspaceToExplore,
                _responseColumnName, _columnNameToIdxMap, theBiggerTheBetter, _buildSpec.te_spec.search_over_columns,  _buildSpec.te_spec.seed);
        break;
    }

    // Pre-setup for grid-based strategies based on AutoML's ways of validating models 
    if(getTeParamsSelectionStrategy() instanceof GridBasedTEParamsSelectionStrategy ) {

      GridBasedTEParamsSelectionStrategy selectionStrategy = (GridBasedTEParamsSelectionStrategy) getTeParamsSelectionStrategy();
      if(_CVEarlyStoppingEnabled) {
        selectionStrategy.setTESearchSpace(ModelValidationMode.CV);
      }
      else {
        selectionStrategy.setTESearchSpace(ModelValidationMode.VALIDATION_FRAME);
      }
    }
    
    _teParams = getTeParamsSelectionStrategy().getBestParams(_modelBuilder);
  }


  public void performAutoTargetEncoding() {

    TargetEncodingParams bestTEParams = getBestTEParams();
    String[] columnsToEncode = bestTEParams.getColumnsToEncode();
    
    Log.info("Best TE parameters were selected to be: columnsToEncode = [ " + StringUtils.join(",", columnsToEncode ) + 
            " ], holdout_type = " + bestTEParams.getHoldoutType() + ", isWithBlending = " + bestTEParams.isWithBlendedAvg() + ", smoothing = " + 
            bestTEParams.getBlendingParams().getF() + ", inflection_point = " + bestTEParams.getBlendingParams().getK() + ", noise_level = " + bestTEParams.getNoiseLevel());
  
    if (columnsToEncode.length > 0) {

      //TODO move it inside TargetEncoder. add constructor ? not all the parameters are used durin
      BlendingParams blendingParams = bestTEParams.getBlendingParams();
      boolean withBlendedAvg = bestTEParams.isWithBlendedAvg();
      boolean imputeNAsWithNewCategory = bestTEParams.isImputeNAsWithNewCategory();
      byte holdoutType = bestTEParams.getHoldoutType();
      double noiseLevel = bestTEParams.getNoiseLevel();
      long seed = _buildSpec.te_spec.seed;

      TargetEncoder tec = new TargetEncoder(columnsToEncode, blendingParams);

      String responseColumnName = _responseColumnName;
      Map<String, Frame> encodingMap = null;

      Frame trainCopy = _trainingFrame.deepCopy(Key.make("train_frame_copy_for_encodings_generation_" + Key.make()).toString());
      DKV.put(trainCopy);

      if (_CVEarlyStoppingEnabled) {
        switch (holdoutType) {
          case TargetEncoder.DataLeakageHandlingStrategy.KFold:
            
            String foldColumnForTE = null;
            foldColumnForTE = _modelBuilder._job._key.toString() + "_te_fold_column";
            int nfolds = 5;
            addKFoldColumn(trainCopy, foldColumnForTE, nfolds, seed);

            encodingMap = tec.prepareEncodingMap(trainCopy, responseColumnName, foldColumnForTE, true);
            Frame encodedTrainingFrame  = tec.applyTargetEncoding(trainCopy, responseColumnName, encodingMap, KFold, foldColumnForTE, withBlendedAvg, noiseLevel, imputeNAsWithNewCategory, seed);
            copyEncodedColumnsToDestinationFrameAndRemoveSource(columnsToEncode, encodedTrainingFrame, _trainingFrame);

            break;
          case TargetEncoder.DataLeakageHandlingStrategy.LeaveOneOut:
          case TargetEncoder.DataLeakageHandlingStrategy.None:
        }
        
        encodingMapCleanUp(encodingMap);
      } else {
        switch (holdoutType) {
          case TargetEncoder.DataLeakageHandlingStrategy.KFold:

            String foldColumnName = getFoldColumnName();
            String autoGeneratedFoldColumnNameForTE = "te_fold_column";

            // 1) If our best TE params, returned from selection strategy, contains DataLeakageHandlingStrategy.KFold as holdoutType
            // then we need kfold column with preferably the same folds as during grid search. 
            // 2) Case when original _trainingFrame does not have fold column. Even with CV enabled we at this point have not yet reached code of folds autoassignments.
            // Best we can do is to add fold column with the same seed as we use during Grid search of TE parameters. Otherwise just apply to the `_foldColumn` from the AutoML setup.
            if(foldColumnName == null) {
              foldColumnName = autoGeneratedFoldColumnNameForTE; // TODO consider introducing config `AutoMLTEControl` keep_te_fold_assignments
              int nfolds = 5; // TODO move to `AutoMLTEControl`
              addKFoldColumn(trainCopy, foldColumnName, nfolds, seed);
            }
            encodingMap = tec.prepareEncodingMap(trainCopy, responseColumnName, foldColumnName, imputeNAsWithNewCategory);

            Frame encodedTrainingFrame = tec.applyTargetEncoding(trainCopy, responseColumnName, encodingMap, holdoutType, foldColumnName, withBlendedAvg, noiseLevel, imputeNAsWithNewCategory, seed);
            copyEncodedColumnsToDestinationFrameAndRemoveSource(columnsToEncode, encodedTrainingFrame, _trainingFrame);

            if(_validationFrame != null) {
              Frame encodedValidationFrame = tec.applyTargetEncoding(_validationFrame, responseColumnName, encodingMap, TargetEncoder.DataLeakageHandlingStrategy.None, foldColumnName, withBlendedAvg, 0, imputeNAsWithNewCategory, seed);
              copyEncodedColumnsToDestinationFrameAndRemoveSource(columnsToEncode, encodedValidationFrame, _validationFrame);
            }
            if(_leaderboardFrame != null) {
              Frame encodedLeaderboardFrame = tec.applyTargetEncoding(_leaderboardFrame, responseColumnName, encodingMap, TargetEncoder.DataLeakageHandlingStrategy.None, foldColumnName, withBlendedAvg, 0, imputeNAsWithNewCategory, seed);
              copyEncodedColumnsToDestinationFrameAndRemoveSource(columnsToEncode, encodedLeaderboardFrame, _leaderboardFrame);
            }
            break;

          case TargetEncoder.DataLeakageHandlingStrategy.LeaveOneOut:
            encodingMap = tec.prepareEncodingMap(trainCopy, responseColumnName, null);

            Frame encodedTrainingFrameLOO = tec.applyTargetEncoding(trainCopy, responseColumnName, encodingMap, holdoutType, withBlendedAvg, noiseLevel, imputeNAsWithNewCategory,seed);
            copyEncodedColumnsToDestinationFrameAndRemoveSource(columnsToEncode, encodedTrainingFrameLOO, _trainingFrame);

            if(_validationFrame != null) {
              Frame encodedValidationFrame = tec.applyTargetEncoding(_validationFrame, responseColumnName, encodingMap, TargetEncoder.DataLeakageHandlingStrategy.None, withBlendedAvg, 0.0,  imputeNAsWithNewCategory, seed);
              copyEncodedColumnsToDestinationFrameAndRemoveSource(columnsToEncode, encodedValidationFrame, _validationFrame);
            }
            if(_leaderboardFrame != null) {
              Frame encodedLeaderboardFrame = tec.applyTargetEncoding(_leaderboardFrame, responseColumnName, encodingMap, TargetEncoder.DataLeakageHandlingStrategy.None, withBlendedAvg, 0.0, imputeNAsWithNewCategory, seed);
              copyEncodedColumnsToDestinationFrameAndRemoveSource(columnsToEncode, encodedLeaderboardFrame, _leaderboardFrame);
            }
            break;
          case TargetEncoder.DataLeakageHandlingStrategy.None:
            //We not only want to search for optimal parameters based on separate test split during grid search but also apply these parameters in the same fashion.
            //But seed is different in current case
            Frame[] trainAndHoldoutSplits = splitByRatio(trainCopy, new double[]{0.7, 0.3}, seed);
            Frame trainNone = trainAndHoldoutSplits[0];
            Frame holdoutNone = trainAndHoldoutSplits[1];
            encodingMap = tec.prepareEncodingMap(holdoutNone, responseColumnName, null);
            Frame encodedTrainingFrameNone = tec.applyTargetEncoding(trainNone, responseColumnName, encodingMap, holdoutType, withBlendedAvg, 0.0, imputeNAsWithNewCategory, seed);
            copyEncodedColumnsToDestinationFrameAndRemoveSource(columnsToEncode, encodedTrainingFrameNone, trainNone);

            _modelBuilder.setTrain(trainNone); 
      
            if(_validationFrame != null) {
              Frame encodedValidationFrameNone = tec.applyTargetEncoding(_validationFrame, responseColumnName, encodingMap, TargetEncoder.DataLeakageHandlingStrategy.None, withBlendedAvg, 0, imputeNAsWithNewCategory, seed);
              copyEncodedColumnsToDestinationFrameAndRemoveSource(columnsToEncode, encodedValidationFrameNone, _validationFrame);
            }
            if(_leaderboardFrame != null) {
              Frame encodedLeaderboardFrameNone = tec.applyTargetEncoding(_leaderboardFrame, responseColumnName, encodingMap, TargetEncoder.DataLeakageHandlingStrategy.None, withBlendedAvg, 0, imputeNAsWithNewCategory, seed);
              copyEncodedColumnsToDestinationFrameAndRemoveSource(columnsToEncode, encodedLeaderboardFrameNone, _leaderboardFrame);
            }
            holdoutNone.delete();
        }
        encodingMapCleanUp(encodingMap);
      }
      trainCopy.delete();
    }

    setColumnsToIgnore(columnsToEncode);

  }
  
  private Frame[] splitByRatio(Frame fr,double[] ratios, long seed) {
    Key[] keys = new Key[]{Key.<Frame>make(), Key.<Frame>make()};
    return ShuffleSplitFrame.shuffleSplitFrame(fr, keys, ratios, seed);
  }

  // Due to TE we want to exclude original column so that we can use only encoded ones. 
  // We need to be careful and reset value in _buildSpec back as next modelBuilder in AutoML sequence will exclude this 
  // columns even before we will have a chance to apply TE.
  void setColumnsToIgnore(String[] columnsToEncode) {
    _buildSpec.input_spec.ignored_columns = concat(_buildSpec.input_spec.ignored_columns, columnsToEncode);
  }

  //Note: we could have avoided this if we were following mutable way in TargetEncoder
  void copyEncodedColumnsToDestinationFrameAndRemoveSource(String[] columnsToEncode, Frame sourceWithEncodings, Frame destinationFrame) {
    for(String column :columnsToEncode) {
      String encodedColumnName = column + "_te";
      Vec encodedVec = sourceWithEncodings.vec(encodedColumnName);
      Vec encodedVecCopy = encodedVec.makeCopy();
      destinationFrame.add(encodedColumnName, encodedVecCopy);
    }
    sourceWithEncodings.delete();
  }

  private void encodingMapCleanUp(Map<String, Frame> encodingMap) {
    if(encodingMap != null) {
      for (Map.Entry<String, Frame> map : encodingMap.entrySet()) {
        map.getValue().delete();
      }
    }
  }

  public static void printOutFrameAsTable(Frame fr) {
    printOutFrameAsTable(fr, false, fr.numRows());
  }

  public static void printOutFrameAsTable(Frame fr, boolean rollups, long limit) {
    assert limit <= Integer.MAX_VALUE;
    TwoDimTable twoDimTable = fr.toTwoDimTable(0, (int) limit, rollups);
    System.out.println(twoDimTable.toString(2, true));
  }

  public TEApplicationStrategy getApplicationStrategy() {
    return _applicationStrategy;
  }


  public String getFoldColumnName() {
    if(_teParams.getHoldoutType() == TargetEncoder.DataLeakageHandlingStrategy.KFold) {
      int foldColumnIndex = _trainingFrame.find(_modelBuilder._parms._fold_column);
      return foldColumnIndex != -1 ? _trainingFrame.name(foldColumnIndex) : null;
    }
    else 
      return null;
  }

  public void setApplicationStrategy(TEApplicationStrategy applicationStrategy) {
    _applicationStrategy = applicationStrategy;
  }

}
