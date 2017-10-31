# visual-speaker-authentication
Visual speaker authentication with random prompt texts by a Multi-task CNN Framework

## 1.prepare data
**put grid lip frames dir in `./data`**
the structure should be like:

```
    ./data/GRID/
    lip/
        s1/
            bbaf2n/
                0.jpeg
                1.jpeg
                2.jpeg
                ...  
                74.jpeg
            ...
        s2/
            ...
        ...
    
    alignments/
        s1/align/
                bbaf2n.align
                ...
        s2/align/
                ...
        ...
```

## 2.train a world model

```
    python train lipnet_res3d.py
```

## 3.train client model

```
    # for example
    # train the client 25's model with his 25 samples 
    python visual_speaker_authentication.py train 24 25
    # test all the other person in the test set. the 0 represent the log file index
    python test 25 0
```
