package hex;

import water.*;
import water.api.schemas3.KeyV3;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;
import water.fvec.Vec;
import water.parser.BufferedString;
import water.util.ArrayUtils;
import water.util.StringUtils;

public class SegmentModels extends Keyed<SegmentModels> {

  private final Frame _segments;
  private final Vec _results;

  public static SegmentModels make(Key<SegmentModels> key, Frame segments) {
    SegmentModels segmentModels = new SegmentModels(key, segments);
    DKV.put(segmentModels);
    return segmentModels;
  }
  
  private SegmentModels(Key<SegmentModels> key, Frame segments) {
    super(key);
    _results = new MakeResultKeys().doAll(Vec.T_STR, segments).outputFrame().vec(0);
    _segments = segments.deepCopy(Key.makeUserHidden(Key.make().toString()).toString());
  }

  SegmentModelResult addResult(long segmentIdx, ModelBuilder mb, Exception e) {
    Key<SegmentModelResult> resultKey = Key.make(_results.atStr(new BufferedString(), segmentIdx).toString());
    SegmentModelResult result = new SegmentModelResult(resultKey, mb._result, getErrors(mb, e), mb._job.warns());
    DKV.put(result);
    return result;
  }
  
  private static String[] getErrors(ModelBuilder mb, Exception e) {
    if (mb.error_count() == 0 && e == null)
      return null;
    String[] errors = new String[0];
    if (mb.error_count() > 0)
      errors = ArrayUtils.append(errors, mb.validationErrors());
    if (e != null)
      errors = ArrayUtils.append(errors, StringUtils.toString(e));
    return errors;
  }
  
  public Frame toFrame() {
    Frame result = new Frame(_segments);
    Frame models = new ToFrame().doAll(3, Vec.T_STR, new Frame(_results))
            .outputFrame(new String[]{"Model ID", "Errors", "Warnings"}, null);
    result.add(models);
    return result;
  }
  
  @Override
  public Class<? extends KeyV3> makeSchema() {
    return KeyV3.SegmentModelsKeyV3.class;
  }

  static class SegmentModelResult extends Keyed<SegmentModelResult> {
    Key<Model> _model;
    String[] _errors;
    String[] _warns;

    SegmentModelResult(Key<SegmentModelResult> k, Key<Model> model, String[] errors, String[] warns) {
      super(k);
      _model = model;
      _errors = errors;
      _warns = warns;
    }
  }

  private static class MakeResultKeys extends MRTask<MakeResultKeys> {
    @Override
    public void map(Chunk c, NewChunk nc) {
      for (int i = 0; i < c._len; i++)
        nc.addStr(Key.makeUserHidden(Key.make().toString()).toString());
    }
  }

  static class ToFrame extends MRTask<ToFrame> {
    @Override
    public void map(Chunk[] cs, NewChunk[] ncs) {
      assert cs.length == 1;
      Chunk c = cs[0];
      BufferedString bs = new BufferedString();
      for (int i = 0; i < c._len; i++) {
        SegmentModelResult result = DKV.getGet(Key.make(c.atStr(bs, i).toString()));
        if (result == null) {
          for (NewChunk nc : ncs)
            nc.addNA();
        } else {
          ncs[0].addStr(result._model.toString());
          if (result._errors != null)
            ncs[1].addStr(String.join("\n", result._errors));
          else
            ncs[1].addNA();
          if (result._warns != null)
            ncs[2].addStr(String.join("\n", result._warns));
          else
            ncs[2].addNA();
        }
      }
    }
  }

  @Override
  protected Futures remove_impl(Futures fs, boolean cascade) {
    if (_segments != null) {
      _segments.remove(fs, cascade);
    }
    if (_results != null) {
      fs.add(new CleanUpSegmentResults().dfork(_results));
    }
    return fs;
  }

  static class CleanUpSegmentResults extends MRTask<CleanUpSegmentResults> {
    @Override
    public void map(Chunk c) {
      BufferedString bs = new BufferedString();
      Futures fs = new Futures();
      for (int i = 0; i < c._len; i++)
        Keyed.remove(Key.make(c.atStr(bs, i).toString()), fs, true);
      fs.blockForPending();
    }
    @Override
    protected void postGlobal() {
      _fr.remove();
    }
  }
  
}
