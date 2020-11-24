To run DAGGER with the expert as the Simulink-exported controller:

```
python dagger.py
```

The actor network weights are automatically saved in the same folder as dagger.py with the name dagger_actor_weight.h5. Additionally, you can save the actor's weights to another path with:

```
python dagger.py --save_path=path_to_save
```

To play the dagger actor:

```
python dagger.py --play --load_path=path_to_load
```