package water.api;

import hex.Model;
import hex.ModelBuilder;
import hex.SegmentModels;
import hex.SegmentModelsBuilder;
import hex.schemas.ModelBuilderSchema;
import water.H2O;
import water.Job;
import water.Key;
import water.TypeMap;
import water.api.schemas3.JobV3;
import water.api.schemas3.ModelParametersSchemaV3;
import water.fvec.Frame;

import java.util.Properties;

public class BulkModelBuilderHandler<B extends ModelBuilder, S extends ModelBuilderSchema<B,S,P>, P extends ModelParametersSchemaV3> extends Handler {

  // Invoke the handler with parameters.  Can throw any exception the called handler can throw.
  @Override
  public JobV3 handle(int version, Route route, Properties parms, String postBody) throws Exception {
    if (! "bulk_train".equals(route._handler_method.getName())) {
      throw new IllegalStateException("Only supports `bulk_train` handler method");
    }

    String segmentsId = parms.getProperty("segments");
    Key<Frame> segmentsKey = Key.make(segmentsId);
    parms.remove("segments");
    Frame segments = segmentsKey.get();

    B builder = makeBuilder(version, route, parms);

    Job<SegmentModels> job = new SegmentModelsBuilder(Key.make(), builder._parms, segments)
            .buildSegmentModels();

    JobV3 schema = new JobV3();
    schema.fillFromImpl(job);
    return schema;
  }

  // FIXME - copy&paste from ModelBuilderHandler
  private B makeBuilder(int version, Route route, Properties parms) {
    // Peek out the desired algo from the URL
    String ss[] = route._url.split("/");
    String algoURLName = ss[3]; // {}/{3}/{ModelBuilders}/{gbm}/{parameters}
    String algoName = ModelBuilder.algoName(algoURLName); // gbm -> GBM; deeplearning -> DeepLearning
    String schemaDir = ModelBuilder.schemaDirectory(algoURLName);

    // Build a Model Schema and a ModelParameters Schema
    String schemaName = schemaDir + algoName + "V" + version;
    S schema = (S) TypeMap.newFreezable(schemaName);
    schema.init_meta();
    String parmName = schemaDir + algoName + "V" + version + "$" + algoName + "ParametersV" + version;
    P parmSchema = (P) TypeMap.newFreezable(parmName);
    schema.parameters = parmSchema;

    Key<Model> key = ModelBuilder.defaultKey(algoName);
    // Default Job for just this training
    Job job = new Job<>(key, ModelBuilder.javaName(algoURLName), algoName);
    // ModelBuilder
    B builder = ModelBuilder.make(algoURLName, job, key);

    schema.parameters.fillFromImpl(builder._parms); // Defaults for this builder into schema
    schema.parameters.fillFromParms(parms);         // Overwrite with user parms
    schema.parameters.fillImpl(builder._parms);     // Merged parms back over Model.Parameter object
    return builder;
  }

  @SuppressWarnings("unused") // formally required but never actually called because handle() is overridden
  public S bulk_train(int version, S schema) { throw H2O.fail(); }

}
