##INSTRUCTIONS TO RUN THE CODE:


CODE CAN ONLY BE RUN IN AN GPU ENABLED ENVIRONMENT

Required Libraries:
  1. PyTorch  1.1.0
  2. sklearn  0.20.1
  3. pandas   0.24.2
  4. numpy    1.13.3

Now download the Dataset from:  https://github.com/Tariq60/LIAR-PLUS/tree/master/dataset  and keep the files test2.tsv, train2.tsv and val2.tsv inside dataset folder

Now run these commands from root folder:
 1. python load_dataset.py 

##FOR SIX WAY CLASSIFICATION:
   1. python network_1.py , python network_2.py, python network_3.py (These can be run parallelly) (#FOR TRAINING THE NETWORK)
   2. python test.py   (#FOR TESTING)


##FOR BINARY CLASSIFICATION:
   1. python network_binary_1.py , python network_binary_2.py, python network_binary_3.py (These can be run parallelly) (#FOR TRAINING THE NETWORK)
   2. python test_binary.py   (#FOR TESTING)
    
   
