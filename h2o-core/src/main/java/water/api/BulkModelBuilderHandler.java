package water.api;

import hex.ModelBuilder;
import hex.SegmentModels;
import hex.SegmentModelsBuilder;
import hex.schemas.ModelBuilderSchema;
import water.H2O;
import water.Job;
import water.Key;
import water.api.schemas3.JobV3;
import water.api.schemas3.ModelParametersSchemaV3;
import water.fvec.Frame;

import java.util.Properties;

public class BulkModelBuilderHandler<B extends ModelBuilder, S extends ModelBuilderSchema<B,S,P>, P extends ModelParametersSchemaV3> extends Handler {

  // Invoke the handler with parameters.  Can throw any exception the called handler can throw.
  @Override
  public JobV3 handle(int version, Route route, Properties parms, String postBody) {
    if (! "bulk_train".equals(route._handler_method.getName())) {
      throw new IllegalStateException("Only supports `bulk_train` handler method");
    }

    String segmentsId = parms.getProperty("segments");
    Key<Frame> segmentsKey = Key.make(segmentsId);
    parms.remove("segments");
    Frame segments = segmentsKey.get();

    final String algoURLName = ModelBuilderHandlerUtils.parseAlgoURLName(route);
    B builder = ModelBuilder.make(algoURLName, null, null);
    ModelBuilderHandlerUtils.makeBuilderSchema(version, algoURLName, parms, builder);

    Job<SegmentModels> job = new SegmentModelsBuilder(Key.make(), builder._parms, segments)
            .buildSegmentModels();

    JobV3 schema = new JobV3();
    schema.fillFromImpl(job);
    return schema;
  }

  @SuppressWarnings("unused") // formally required but never actually called because handle() is overridden
  public S bulk_train(int version, S schema) { throw H2O.fail(); }

}
