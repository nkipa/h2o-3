package water.api;

import water.*;
import water.api.schemas3.KeyV3;
import water.api.schemas3.RemoveAllV3;
import water.util.Log;

// Best-effort cluster brain-wipe and reset.
// Useful between unrelated tests.
public class RemoveAllHandler extends Handler {
  @SuppressWarnings("unused") // called through reflection by RequestServer
  public RemoveAllV3 remove(int version, RemoveAllV3 u) {
    Futures fs = new Futures();
    // Cancel and remove leftover running jobs
    for( Job j : Job.jobs() ) { j.stop_requested(); j.remove(fs); }
    // Wipe out any and all session info
    if( RapidsHandler.SESSIONS != null ) {
      for(String k: RapidsHandler.SESSIONS.keySet() )
        (RapidsHandler.SESSIONS.get(k)).endQuietly(null);
      RapidsHandler.SESSIONS.clear();
    }
    fs.blockForPending();

    if (u.retained_keys != null && u.retained_keys.length != 0) {
      retainKeys(u.retained_keys);
    } else {
      clearAll();
    }
    Log.info("Finished removing objects");
    return u;
  }


  private void clearAll() {
    Log.info("Removing all objects");
    // Bulk brainless key removal.  Completely wipes all Keys without regard.
    new MRTask(H2O.MIN_HI_PRIORITY) {
      @Override
      public void setupLocal() {
        H2O.raw_clear();
        water.fvec.Vec.ESPC.clear();
      }
    }.doAllNodes();
    // Wipe the backing store without regard as well
    H2O.getPM().getIce().cleanUp();
    H2O.updateNotIdle();
  }

  private void retainKeys(final KeyV3[] retained_keys) {
    Log.info(String.format("Removing all objects, except for %d provided key(s)", retained_keys.length));
    final Key[] retainedKeys;
    if (retained_keys == null) {
      retainedKeys = new Key[0];
    } else {
      retainedKeys = new Key[retained_keys.length];
      for (int i = 0; i < retainedKeys.length; i++) {
        if (retained_keys[i] == null) continue; // Protection against null keys from the client - ignored
        retainedKeys[i] = retained_keys[i].key();
      }
    }

    new DKV.ClearDKVTask(retainedKeys)
            .doAllNodes();

  }
}
