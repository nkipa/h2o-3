package hex.gam;

import hex.gam.GAMModel.GAMParameters.BSType;
import hex.glm.GLMModel;
import org.junit.BeforeClass;
import org.junit.Test;
import water.DKV;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;

/***
 * Here I am going to test the following:
 * - model matrix formation with centering
 */
public class GamTestPiping extends TestUtil {
  @BeforeClass
  public static void setup() {
    stall_till_cloudsize(1);
  }
  
  @Test
  public void testAdaptFrame() {
    try {
      Scope.enter();
      Frame train = parse_test_file("./smalldata/gam_test/gamDataRegressionOneFun.csv");
      Scope.track(train);
      Frame trainCorrectOutput = parse_test_file("./smalldata/gam_test/gamDataRModelMatrixCenterDataOneFun.csv");
      Scope.track(trainCorrectOutput);
      int numKnots = 6;
      double hj = (train.vec(1).max()-train.vec(1).min())/(numKnots-1);
      double oneOhj = 1.0/hj;
      double[][] matD = new double[numKnots-2][];
      matD[0] = new double[]{oneOhj, -2*oneOhj, oneOhj, 0, 0, 0};
      matD[1] = new double[]{0, oneOhj, -2*oneOhj, oneOhj, 0, 0};
      matD[2] = new double[]{0, 0, oneOhj, -2*oneOhj, oneOhj, 0};
      matD[3] = new double[]{0, 0, 0, oneOhj, -2*oneOhj, oneOhj};
      double[][] matB = new double[numKnots-2][];
      matB[0] = new double[]{2*hj/3, hj/6, 0, 0};
      matB[1] = new double[]{hj/6, 2*hj/3, hj/6,0};
      matB[2] = new double[]{0, hj/6, 2*hj/3, hj/6};
      matB[3] = new double[]{0, 0, hj/6, 2*hj/3};
      double[][] bInvD = new double[numKnots-2][];
      bInvD[0] = new double[]{4.019621461444700e+01, -9.115927242919231e+01, 6.460105920178982e+01, 
              -1.722694912047729e+01, 4.306737280119322e+00, -7.177895466865537e-01};
      bInvD[1] = new double[]{-1.076684320029831e+01, 6.460105920178982e+01, -1.083862215496696e+02, 
              6.890779648190914e+01, -1.722694912047729e+01, 2.871158186746215e+00};
      bInvD[2] = new double[]{2.871158186746215e+00, -1.722694912047729e+01,  6.890779648190914e+01,
              -1.083862215496696e+02, 6.460105920178982e+01, -1.076684320029830e+01};
      bInvD[3] = new double[]{-7.177895466865537e-01, 4.306737280119322e+00, -1.722694912047729e+01,
              6.460105920178982e+01, -9.115927242919230e+01, 4.019621461444699e+01};
/*      Frame predictVec = new Frame(train.vec(1));
      GenerateGamMatrixOneColumn oneAugCol = new GenerateGamMatrixOneColumn(BSType.cr, numKnots, null, predictVec,
              false).doAll(numKnots, Vec.T_NUM, predictVec);
      Frame oneAugmentedColumn = oneAugCol.outputFrame(null, null, null);*/
      
      GAMModel.GAMParameters parms = new GAMModel.GAMParameters();
      parms._bs = new BSType[]{BSType.cr};
      parms._k = new int[]{6};
      parms._response_column = train.name(2);
      parms._ignored_columns = new String[]{train.name(0), train.name(1)}; // row of ids
      parms._gam_X = new String[]{train.name(1)};
      parms._train = train._key;
      parms._family = GLMModel.GLMParameters.Family.gaussian;
      parms._link = GLMModel.GLMParameters.Link.family_default;

      GAMModel model = new GAM(parms).trainModel().get();
      Scope.track_generic(model);
    } finally {
      Scope.exit();
    }
  }

  @Test
  public void testAdaptFrame2GAMColumns() {
    try {
      Scope.enter();
      Frame train = parse_test_file("./smalldata/gam_test/gamDataRegressionTwoFuns.csv");
      Scope.track(train);
      GAMModel.GAMParameters parms = new GAMModel.GAMParameters();
      parms._bs = new BSType[]{BSType.cr, BSType.cr};
      parms._k = new int[]{6,6};
      parms._response_column = train.name(3);
      parms._ignored_columns = new String[]{train.name(0), train.name(1), train.name(2)}; // row of ids
      parms._gam_X = new String[]{train.name(1), train.name(2)};
      parms._train = train._key;
      parms._family = GLMModel.GLMParameters.Family.gaussian;
      parms._link = GLMModel.GLMParameters.Link.family_default;
      parms._saveZMatrix = true;
      parms._saveGamCols = true;

      GAMModel model = new GAM(parms).trainModel().get();
      Frame transformedData = ((Frame) DKV.getGet(model._output._gamTransformedTrain));
      Scope.track(transformedData);
      Frame predictF = Scope.track(model.score(train)); // predict with train data
      Scope.track(predictF);
      System.out.println("Wow");
      Scope.track_generic(model);
    } finally {
      Scope.exit();
    }
  }

  @Test
  public void testGAMGaussian() {
    try {
      Scope.enter();
      Frame train = parse_test_file("./smalldata/glm_test/gaussian_20cols_10000Rows.csv");
      int numCols = train.numCols();
      int enumCols = (numCols-1)/2;
      for (int cindex=0; cindex<enumCols; cindex++) {
        train.replace(cindex, train.vec(cindex).toCategoricalVec()).remove();
      }
      String[] ignoredCols = new String[]{"C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", 
              "C15", "C16", "C17", "C18", "C19", "C20"};
      String[] gamCols = new String[]{"C11", "C12"};
      DKV.put(train);
      Scope.track(train);
      
      GAMModel.GAMParameters parms = new GAMModel.GAMParameters();
      parms._bs = new BSType[]{BSType.cr, BSType.cr};
      parms._k = new int[]{6,6};
      parms._response_column = "C21";
      parms._ignored_columns = ignoredCols;
      parms._gam_X = gamCols;
      parms._train = train._key;
      parms._family = GLMModel.GLMParameters.Family.gaussian;
      parms._link = GLMModel.GLMParameters.Link.family_default;
      parms._saveZMatrix = true;
      parms._saveGamCols = true;
      parms._standardize = true;

      GAMModel model = new GAM(parms).trainModel().get();
      Frame transformedData = ((Frame) DKV.getGet(model._output._gamTransformedTrain));
      Scope.track(transformedData);
/*      Frame predictF = Scope.track(model.score(train)); // predict with train data
      Scope.track(predictF);*/
      System.out.println("Wow");
      Scope.track_generic(model);
    } finally {
      Scope.exit();
    }
  }

  @Test
  public void testStandardizedCoeff() {
    String[] ignoredCols = new String[]{"C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", 
            "C17", "C18", "C19", "C20"};
    String[] gamCols = new String[]{"C11", "C12"};
    // test for Gaussian
    testCoeffs(GLMModel.GLMParameters.Family.gaussian, "smalldata/glm_test/gaussian_20cols_10000Rows.csv",
            "C21", gamCols, ignoredCols);
    // test for binomial
    testCoeffs(GLMModel.GLMParameters.Family.binomial, "smalldata/glm_test/binomial_20_cols_10KRows.csv", 
            "C21", gamCols, ignoredCols);
    // test for multinomial
    ignoredCols = new String[]{"C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
    gamCols = new String[]{"C6", "C7"};
    testCoeffs(GLMModel.GLMParameters.Family.multinomial,
            "smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv", "C11",
            gamCols, ignoredCols);

  }

  public void testCoeffs(GLMModel.GLMParameters.Family family, String fileName, String responseColumn, 
                         String[] gamCols, String[] ignoredCols) {
    try {
      Scope.enter();
      Frame train = parse_test_file(fileName);
      // set cat columns
      int numCols = train.numCols();
      int enumCols = (numCols-1)/2;
      for (int cindex=0; cindex<enumCols; cindex++) {
        train.replace(cindex, train.vec(cindex).toCategoricalVec()).remove();
      }
      int response_index = numCols-1;
      if (family.equals(GLMModel.GLMParameters.Family.binomial) || (family.equals(GLMModel.GLMParameters.Family.multinomial))) {
        train.replace((response_index), train.vec(response_index).toCategoricalVec()).remove();
      }
      DKV.put(train);
      Scope.track(train);

      GAMModel.GAMParameters params = new GAMModel.GAMParameters();
      params._standardize=false;
      params._family = family;
      params._response_column = responseColumn;
      params._train = train._key;
      params._bs = new BSType[]{BSType.cr, BSType.cr};
      params._k = new int[]{6,6};
      params._ignored_columns = ignoredCols;
      params._gam_X = gamCols;
      params._train = train._key;
      params._family = family;
      params._link = GLMModel.GLMParameters.Link.family_default;
      params._saveZMatrix = true;
      params._saveGamCols = true;
      params._standardize = true;
      GAMModel gam = new GAM(params).trainModel().get();
      Scope.track_generic(gam);
      Frame transformedData = ((Frame) DKV.getGet(gam._output._gamTransformedTrain));
      Scope.track(transformedData);
      numCols = transformedData.numCols()-1;
      for (int ind = 0; ind < numCols; ind++)
        System.out.println(transformedData.vec(ind).mean());
      Frame transformedDataCenter = ((Frame) DKV.getGet(gam._output._gamTransformedTrainCenter));
      Scope.track(transformedDataCenter);
      numCols = transformedDataCenter.numCols()-1;
      System.out.println("Print center gamx");
      for (int ind = 0; ind < numCols; ind++)
        System.out.println(transformedDataCenter.vec(ind).mean());
      Frame predictF = gam.score(transformedData); // predict with train data
      Scope.track(predictF);
      Frame predictRaw = gam.score(train); // predict with train data
      Scope.track(predictRaw);
      TestUtil.assertIdenticalUpToRelTolerance(predictF, predictRaw, 1e-6);
    } finally {
      Scope.exit();
    }
  }


}
