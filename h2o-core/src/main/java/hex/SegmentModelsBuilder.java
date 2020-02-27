package hex;

import water.*;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;
import water.fvec.Vec;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class SegmentModelsBuilder {

  private final Key<SegmentModels> _dest;
  private final Model.Parameters _blueprintParms;
  private final Frame _segments;

  public SegmentModelsBuilder(Key<SegmentModels> dest, Model.Parameters blueprintParms, Frame segments) {
    _dest = dest;
    _blueprintParms = blueprintParms;
    _segments = validateSegmentsFrame(segments);
  }

  private static Frame validateSegmentsFrame(Frame segments) {
    List<String> invalidColumns = Stream.of(segments.names())
            .filter(name -> !segments.vec(name).isCategorical() && !segments.vec(name).isInt())
            .collect(Collectors.toList());
    if (!invalidColumns.isEmpty()) {
      throw new IllegalStateException(
              "Columns to segment-by can only be categorical and integer of type, invalid columns: " + invalidColumns);
    }
    return segments;
  }

  public Job<SegmentModels> buildSegmentModels() {
    Job<SegmentModels> job = new Job<>(_dest, SegmentModels.class.getName(), _blueprintParms.algoName());
    SegmentModeledBuilderTask segmentBuilder = new SegmentModeledBuilderTask(
            job, _blueprintParms._train, _blueprintParms._valid);
    return job.start(segmentBuilder, _segments.numRows());
  }

  private class SegmentModeledBuilderTask extends H2O.H2OCountedCompleter<SegmentModeledBuilderTask> {
    private final Job<SegmentModels> _job;
    private final Frame _full_train;
    private final Frame _full_valid;

    private SegmentModeledBuilderTask(Job<SegmentModels> job, Key<Frame> train, Key<Frame> valid) {
      _job = job;
      _full_train = reorderColumns(train);
      _full_valid = reorderColumns(valid);
    }

    @Override
    public void compute2() {
      try {
        _blueprintParms.read_lock_frames(_job);
        SegmentModels segmentModels = SegmentModels.make(_dest, _segments);
        buildModels(segmentModels);
      } finally {
        _blueprintParms.read_unlock_frames(_job);
      }
      tryComplete();
    }
    
    void buildModels(SegmentModels segmentModels) {
      Vec.Reader[] segmentVecReaders = new Vec.Reader[_segments.numCols()];
      for (int i = 0; i < segmentVecReaders.length; i++)
        segmentVecReaders[i] = _segments.vec(i).new Reader();
      for (long segmentIdx = 0; segmentIdx < _segments.numRows(); segmentIdx++) {
        if (_job.stop_requested())
          throw new Job.JobCancelledException();  // Handle end-user cancel request

        double[] segmentVals = readRow(segmentVecReaders, segmentIdx);
        final ModelBuilder builder = makeBuilder(segmentIdx, segmentVals);

        Exception failure = null;
        try {
          builder.init(false);
          if (builder.error_count() == 0)
            builder.trainModelOnH2ONode().get();
        } catch (Exception e) {
          failure = e;
        } finally {
          _job.update(1);
          segmentModels.addResult(segmentIdx, builder, failure);
          cleanUp(builder);
        }
      }
    }

    private void cleanUp(ModelBuilder builder) {
      Futures fs = new Futures();
      Keyed.remove(builder._parms._train, fs, true);
      Keyed.remove(builder._parms._valid, fs, true);
      fs.blockForPending();
    }
    
    private ModelBuilder makeBuilder(long segmentIdx, double[] segmentVals) {
      ModelBuilder builder = ModelBuilder.make(_blueprintParms);
      builder._parms._train = makeSegmentFrame(_full_train, segmentIdx, segmentVals);
      builder._parms._valid = makeSegmentFrame(_full_valid, segmentIdx, segmentVals);
      return builder;
    }
    
    private Key<Frame> makeSegmentFrame(Frame f, long segmentIdx, double[] segmentVals) {
      if (f == null)
        return null;
      Key<Frame> segmentFrameKey = Key.make(f.toString() + "_segment_" + segmentIdx);
      Frame segmentFrame = new MakeSegmentFrame(segmentVals)
              .doAll(f.types(), f)
              .outputFrame(segmentFrameKey, f.names(), f.domains());
      assert segmentFrameKey.equals(segmentFrame._key);
      return segmentFrameKey;
    }
    
    private Frame reorderColumns(Key<Frame> key) {
      if (key == null)
        return null;
      Frame f = key.get();
      if (f == null) {
        throw new IllegalStateException("Key " + key + " doesn't point to an existing Frame.");
      }
      Frame mutating = new Frame(f);
      Frame reordered = new Frame(_segments.names(), mutating.vecs(_segments.names()))
              .add(mutating.remove(_segments.names()));
      reordered._key = f._key;
      return reordered;
    }

  }

  private static class MakeSegmentFrame extends MRTask<MakeSegmentFrame> {
    private final double[] _match_row;

    MakeSegmentFrame(double[] matchRow) {
      _match_row = matchRow;
    }

    @Override
    public void map(Chunk[] cs, NewChunk[] ncs) {
      int cnt = 0;
      int rows[] = new int[cs[0]._len];
      each_row: for (int row = 0; row < rows.length; row++) {
        for (int i = 0; i < _match_row.length; i++) {
          if (Double.isNaN(_match_row[i]) && !cs[i].isNA(row))
            continue each_row;
          if (_match_row[i] != cs[i].atd(row))
            continue each_row;
        }
        rows[cnt++] = row; 
      }
      if (cnt == 0)
        return;
      rows = cnt == rows.length ? rows : Arrays.copyOf(rows, cnt);
      for (int i = 0; i < cs.length; i++) {
        cs[i].extractRows(ncs[i], rows);
      }
    }
  }

  private static double[] readRow(Vec.Reader[] vecReaders, long r) {
    double[] row = new double[vecReaders.length];
    for (int i = 0; i < row.length; i++)
      row[i] = vecReaders[i].isNA(r) ? Double.NaN : vecReaders[i].at(r);
    return row;
  }

}
