package hex.gam;

import hex.DataInfo;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.gam.GAMModel.GAMParameters;
import hex.gam.MatrixFrameUtils.GamUtils;
import hex.glm.GLM;
import hex.glm.GLMModel;
import hex.glm.GLMModel.GLMParameters;
import water.DKV;
import water.Key;
import water.MemoryManager;
import water.Scope;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static hex.gam.MatrixFrameUtils.GamUtils.*;


public class GAM extends ModelBuilder<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{ModelCategory.Regression};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Experimental;
  }

  @Override
  public boolean havePojo() {
    return false;
  }

  @Override
  public boolean haveMojo() {
    return false;
  }

  public GAM(boolean startup_once) {
    super(new GAMModel.GAMParameters(), startup_once);
  }

  public GAM(GAMModel.GAMParameters parms) {
    super(parms);
    init(false);
  }

  public GAM(GAMModel.GAMParameters parms, Key<GAMModel> key) {
    super(parms, key);
    init(false);
  }
  
  @Override
  public void init(boolean expensive) {
    super.init(expensive);
    if (expensive) {  // add custom check here
      if (error_count() > 0)
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
      
      if (_parms._gam_X == null)
        error("_gam_X", "must specify columns indices to apply GAM to.  If you don't have any, use GLM.");
      if (_parms._k == null) {  // user did not specify any knots, we will use default 10, evenly spread over whole range
        int numKnots = _train.numRows() < 10 ? (int) _train.numRows() : 10;
        _parms._k = new int[_parms._gam_X.length];  // different columns may have different 
        Arrays.fill(_parms._k, numKnots);
      }
      if ((_parms._saveGamCols || _parms._saveZMatrix) && ((_train.numCols() - 1 + _parms._k.length) < 2))
        error("_saveGamCols/_saveZMatrix", "can only be enabled if we number of predictors plus" +
                " Gam columns in _gamX exceeds 2");
      if ((_parms._lambda_search || !_parms._intercept || _parms._lambda == null || _parms._lambda[0] > 0)) 
        _parms._use_all_factor_levels = true;
      if (_parms._link.equals(GLMParameters.Link.family_default))
        _parms._link = _parms._family.defaultLink;
    }
  }

  @Override
  protected boolean computePriorClassDistribution() {
    return (_parms._family== GLMParameters.Family.multinomial)||(_parms._family== GLMParameters.Family.ordinal);
  }
  
  @Override
  protected GAMDriver trainModelImpl() {
    return new GAMDriver();
  }

  @Override
  protected int nModelsInParallel(int folds) {
    return nModelsInParallel(folds, 2);
  }

  @Override
  protected void checkMemoryFootPrint_impl() {
    ;
  }

  private class GAMDriver extends Driver {
    boolean _centerGAM = false; // true if we need to constraint GAM columns
    double[][][] _zTranspose; // store for each GAM predictor transpose(Z) matrix
    double[][][] _penalty_mat;  // store for each GAM predictir the penalty matrix
    public double[][][] _binvD; // store BinvD for each gam column specified for scoring
    public double[][] _knots; // store knots location for each gam column
    public int[] _numKnots;  // store number of knots per gam column
    String[][] _gamColNames;  // store column names of GAM columns
    String[][] _gamColNamesCenter;  // gamColNames after de-centering is performed.
    Key<Frame>[] _gamFrameKeys;
    Key<Frame>[] _gamFrameKeysCenter;

    /***
     * This method will take the _train that contains the predictor columns and response columns only and add to it
     * the following:
     * 1. For each predictor included in gam_x, expand it out to calculate the f(x) and attach to the frame.
     * 2. It will calculate the ztranspose that is used to center the gam columns.
     * 3. It will calculate a penalty matrix used to control the smoothness of GAM.
     * 
     * @return
     */
    Frame adaptTrain() {
      int numGamFrame = _parms._gam_X.length;
      _centerGAM = (numGamFrame > 1) || (_train.numCols() - 1 + numGamFrame) >= 2;
      _zTranspose = _centerGAM ? GamUtils.allocate3DArray(numGamFrame, _parms, 0) : null;
      _penalty_mat = _centerGAM ? GamUtils.allocate3DArray(numGamFrame, _parms, 2) :
              GamUtils.allocate3DArray(numGamFrame, _parms, 1);
      _binvD = GamUtils.allocate3DArray(numGamFrame, _parms, 3);
      _numKnots = MemoryManager.malloc4(numGamFrame);
      _knots = new double[numGamFrame][];
      _gamColNames = new String[numGamFrame][];
      _gamColNamesCenter = new String[numGamFrame][];
      _gamFrameKeys = new Key[numGamFrame];
      _gamFrameKeysCenter = new Key[numGamFrame];

      addGAM2Train(_parms, _parms.train(), _zTranspose, _penalty_mat, _gamColNames, _gamColNamesCenter,
              true, _centerGAM, _gamFrameKeys, _gamFrameKeysCenter, _binvD, _numKnots, _knots);
      return buildGamFrame(numGamFrame, _gamFrameKeys, _train, _parms._response_column); // add gam cols to _train
    }

    @Override
    public void computeImpl() {
      init(true);     //this can change the seed if it was set to -1
      if (error_count() > 0)   // if something goes wrong, let's throw a fit
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);

      _job.update(0, "Initializing model training");

      buildModel(); // build gam model 
    }

    public final void buildModel() {
      GAMModel model = null;
      DataInfo dinfo = null;
      Frame newTFrame=null, gamColsFrame=null, gamColsFrameCenter=null;
      try {
        _job.update(0, "Adding GAM columns to training dataset...");
        int trainCols = _train.numCols();  // number of columns before adding GAM columns
        String[] trainColNames = _train.names(); // column names before adding GAM columns
        Frame newTrain = rebalance(adaptTrain(), false, _result+".temporary.train"); // add and store gam cols without centering
        dinfo = new DataInfo(newTrain.clone(), _valid, 1, _parms._use_all_factor_levels || _parms._lambda_search, DataInfo.TransformType.NONE, DataInfo.TransformType.NONE,
                _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.Skip,
                _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.MeanImputation || _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.PlugValues,
                _parms.makeImputer(),
                false, hasWeightCol(), hasOffsetCol(), hasFoldCol(), null);
        DKV.put(dinfo._key, dinfo);
        model = new GAMModel(dest(), _parms, new GAMModel.GAMModelOutput(GAM.this, dinfo._adaptedFrame, dinfo));
        model.delete_and_lock(_job);
        newTFrame = _centerGAM?new Frame(buildGamFrameCenter(_parms._gam_X.length, _gamFrameKeysCenter, _parms.train(), 
                _parms._response_column, _parms.train().numCols(), _parms.train().names())):new Frame(dinfo._adaptedFrame);  // get frames with correct predictors and spline functions
        DKV.put(newTFrame); // This one will cause deleted vectors if add to Scope.track
        if (_parms._saveGamCols) {  // save gam column keys
          gamColsFrame = saveGAMFrames(newTrain);
          DKV.put(gamColsFrame);  // do not Scope.track this, will cause null frame
          model._output._gamTransformedTrain = gamColsFrame._key;
          if (_centerGAM) {
            gamColsFrameCenter = saveGAMFrames(newTFrame);
            DKV.put(gamColsFrameCenter);
            model._output._gamTransformedTrainCenter = gamColsFrameCenter._key;
            model._output._gamGamXCenter = newTFrame._key;
          }
        }
        
        _job.update(1, "calling GLM to build GAM model...");
        GLMModel glmModel = buildGLMModel(_parms, newTFrame); // obtained GLM model
        Scope.track_generic(glmModel);
        _job.update(0, "Building out GAM model...");
        fillOutGAMModel(glmModel, model, dinfo); // build up GAM model 
        
        // call adaptTeatForTrain() to massage frame before scoring.
        // call model.makeModelMetrics
        // create model summary by calling createModelSummaryTable or something like that
        model.update(_job);
        // build GAM Model Metrics
      } finally {
        List<Key<Vec>> keep = new ArrayList<>();
        if (model != null) {
          if (_parms._saveGamCols) {
            addFrameKeys2Keep(newTFrame._key, keep);
            addFrameKeys2Keep(model._output._gamTransformedTrain, keep);
            if (_centerGAM)
              addFrameKeys2Keep(model._output._gamTransformedTrainCenter, keep);
          }
          model.unlock(_job);
        }
        if (dinfo!=null)
          dinfo.remove();
        Scope.untrack(keep);  // leave the vectors alone.
      }
    }
    
    GLMModel buildGLMModel(GAMParameters parms, Frame trainData) {
      GLMParameters glmParam = GamUtils.copyGAMParams2GLMParams(parms, trainData);  // copy parameter from GAM to GLM
      GLMModel model = new GLM(glmParam, _penalty_mat, _centerGAM? _gamColNamesCenter : _gamColNames).trainModel().get();
      return model;
    }
    
    void fillOutGAMModel(GLMModel glm, GAMModel model, DataInfo dinfo) {
      model._centerGAM = _centerGAM;
      model._gamColNames = _gamColNames;  // copy over gam column names
      model._gamColNamesCenter = _gamColNamesCenter;
      model._output._zTranspose = _zTranspose;
      model._gamFrameKeys = _gamFrameKeys;
      model._gamFrameKeysCenter = _gamFrameKeysCenter;
      model._nclass = _nclass;
      model._output._binvD = _binvD;
      model._output._knots = _knots;
      model._output._numKnots = _numKnots;
      if (_parms._savePenaltyMat)
        model._output._penaltyMatrices = _penalty_mat;
      copyGLMCoeffs(glm, model, dinfo);  // copy over coefficient names and generate coefficients as beta = z*GLM_beta
    }
    
    void copyGLMCoeffs(GLMModel glm, GAMModel model, DataInfo dinfo) {
      int totCoeffNums = dinfo.fullN()+1;
      model._output._coefficient_names = new String[totCoeffNums]; // copy coefficient names from GLM to GAM
      int gamNumStart = copyGLMCoeffNames2GAMCoeffNames(model, glm, dinfo);
      copyGLMCoeffs2GAMCoeffs(model, glm, dinfo, _parms._family, gamNumStart, _parms._standardize, _nclass); // obtain beta without centering
    }
  }
}
