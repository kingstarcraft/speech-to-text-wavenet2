# Speech-to-Text-WaveNet2 : End-to-end sentence level English speech recognition using DeepMind's WaveNet
A tensorflow implementation of speech recognition based on DeepMind's [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499). (Hereafter the Paper)

The architecture is shown in the following figure.
<p align="center">
  <img src="https://raw.githubusercontent.com/buriburisuri/speech-to-text-wavenet/master/png/architecture.png" width="1024"/>
</p>
(Some images are cropped from [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) and [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099))  


## Version 

Current Version : __***2.0.0.0***__
- [x] demo
- [x] test
- [ ] train

## Dependencies

1. tensorflow (passed with 1.12.0)
1. librosa (passed with 0.1)
1. glog (passed with 0.3.1)

If you have problems with the librosa library, try to install ffmpeg by the following command. ( Ubuntu 14.04 )  
<pre><code>
sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update
sudo apt-get dist-upgrade -y
sudo apt-get -y install ffmpeg
</code></pre>

## Dataset

We used [VCTK](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html), 
[LibriSpeech](http://www.openslr.org/12/) and [TEDLIUM release 2](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus) corpus.
Total number of sentences in the training set composed of the above three corpus is 240,612. 
Valid and test set is built using only LibriSpeech and TEDLIUM corpuse, because VCTK corpus does not have valid and test set. 
After downloading the each corpus, extract them in the 'asset/data/VCTK-Corpus', 'asset/data/LibriSpeech' and 
 'asset/data/TEDLIUM_release2' directories. 
 
Audio was augmented by the scheme in the [Tom Ko et al](http://speak.clsp.jhu.edu/uploads/publications/papers/1050_pdf.pdf)'s paper. 
(Thanks @migvel for your kind information)  

## Pre-processing dataset

- Download and extract dataset(only VCTK support now.)
- Assume the directory of VCTK dataset is f:/speech, Execute
```
python create_tf_record.py -input_dir='f:/speech'
```
to create record for train or test

## Training the network
Coming soon !

Execute
<pre><code>
python train.py ( <== Use all available GPUs )
or
CUDA_VISIBLE_DEVICES=0,1 python train.py ( <== Use only GPU 0, 1 )
</code></pre>
to train the network. You can see the result ckpt files and log files in the 'asset/train' directory.
Launch tensorboard --logdir asset/train/log to monitor training process.

We've trained this model on a 3 Nvidia 1080 Pascal GPUs during 40 hours until 50 epochs and we picked the epoch when the 
validatation loss is minimum. In our case, it is epoch 40.  If you face the out of memory error, 
reduce batch_size in the train.py file from 16 to 4.  

The CTC losses at each epoch are as following table:

| epoch | train set | valid set | test set | 
| :----: | ----: | ----: | ----: |
| 20 | 79.541500 | 73.645237 | 83.607269 |
| 30 | 72.884180 | 69.738348 | 80.145867 |
| 40 | 69.948266 | 66.834316 | 77.316114 |
| 50 | 69.127240 | 67.639895 | 77.866674 |


## Testing the network
After training finished, you can evalute mode by the following command.
```
python test.py
```

## Transforming speech wave file to English text 
 
Execute
<pre><code>
python demo.py -input_path <wave_file path>
</code></pre>
to transform a speech wave file to the English sentence. The result will be printed on the console. 

For example, try the following command.
<pre><code>
python demo.py -input_path=data/demo.wav -ckpt_dir=pretrained
</code></pre>

The result will be as follows:
<pre><code>
please scool stella
</code></pre>

The ground truth is as follows:
<pre><code>
PLEASE SCOOL STELLA
</code></pre>

As mentioned earlier, there is no language model, so there are some cases where capital letters, punctuations, and words are misspelled.

## pre-trained models

You can transform a speech wave file to English text with the pre-trained model on the VCTK corpus. 
Extract [the following zip file](https://drive.google.com/file/d/1K2PDW4pEH_ieX0WLQvv1aA_iKsr6xle4/view?usp=sharing) to the 'model' directory, and make sure the config.json in 'model/buriburisuri'.

## Docker support

See docker [README.md](docker/README.md).

## Future works

1. Language Model
1. Polyglot(Multi-lingual) Model

We think that we should replace CTC beam decoder with a practical language model  
and the polyglot speech recognition model will be a good candidate to future works.

## Other resources

1. [buriburisuri's speech-to-text-wavenet](https://github.com/buriburisuri/speech-to-text-wavenet.git)
1. [ibab's WaveNet(speech synthesis) tensorflow implementation](https://github.com/ibab/tensorflow-wavenet)
1. [tomlepaine's Fast WaveNet(speech synthesis) tensorflow implementation](https://github.com/ibab/tensorflow-wavenet)

## Namju's other repositories

1. [SugarTensor](https://github.com/buriburisuri/sugartensor)
1. [EBGAN tensorflow implementation](https://github.com/buriburisuri/ebgan)
1. [Timeseries gan tensorflow implementation](https://github.com/buriburisuri/timeseries_gan)
1. [Supervised InfoGAN tensorflow implementation](https://github.com/buriburisuri/supervised_infogan)
1. [AC-GAN tensorflow implementation](https://github.com/buriburisuri/ac-gan)
1. [SRGAN tensorflow implementation](https://github.com/buriburisuri/SRGAN)
1. [ByteNet-Fast Neural Machine Translation](https://github.com/buriburisuri/ByteNet)

## Citation

If you find this code useful please cite us in your work:

<pre><code>
Kim and Park. Speech-to-Text-WaveNet. 2016. GitHub repository. https://github.com/buriburisuri/.
</code></pre>

# Authors

Namju Kim (namju.kim@kakaocorp.com) at KakaoBrain Corp.

Kyubyong Park (kbpark@jamonglab.com) at KakaoBrain Corp.
