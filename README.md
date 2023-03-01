#### General Description
This models were trained on AWID2 [1] dataset containing traffic from IEEE 802.11 wireless networks where netowork attacks occured. This models are trained to datect such a threats. 

#### Configuration

All you need for usage of examplary testing script `model_test.py` or loading this models into your own script is to execute such a command to install requirements: 
`pip install -r requirements.txt` 

#### Examplary usage
The examplary testing script is presented in `model_test.py` file. 

#### Loading models into your own Python 3 scripts

If you would like to load this model and test it on your own environment the only thing you need to do is to load it using Tensoflow framework like this:

`model = tf.keras.models.load_model('.cnn_cls_94')`

More detailed information about this model can be found in this paper [2].


References:

1. C. Kolias, G. Kambourakis, A. Stavrou and S. Gritzalis, "Intrusion Detection in 802.11 Networks: Empirical Evaluation of Threats and a Public Dataset," in IEEE Communications Surveys & Tutorials, vol. 18, no. 1, pp. 184-208, Firstquarter 2016, doi: 10.1109/COMST.2015.2402161.
2. Natkaniec, M.; Bednarz, M., Wireless Local Area Networks Threat Detection Using CNN. Submitted to Sensors 2023.