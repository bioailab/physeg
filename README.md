![Examples](img/img.png)

### Publication
[Sekh, Arif Ahmed](https://www.linkedin.com/in/arif1984/), Ida S. Opstad, Gustav Godtliebsen, Åsa Birna Birgisdottir, Balpreet Singh Ahluwalia, Krishna Agarwal, and Dilip K. Prasad. "Physics based machine learning for sub-cellular segmentation in living cells.
"Submitted in Nature Machine Intelligence. 2021. 
[Link to The Paper](https://www.nature.com/articles/s42256-021-00420-0)

<br />


### Description

### Software and Hardware
======================
OS: Windows 10 <br />
GPU: NVIDIA Quadro RTX 6000 <br />
Anaconda 64 bit with Python 3.6.8 <br />
Matlab 2019b <br />


### Create conda environment with all required packages
conda env create --name segment --file=segment.yml<br />
activate segment<br />
OR<br />
install all dependency listed in segment.yml <br />


### Install Matlab Engine for Python

1. Installing the MatLab library <br />

Following the instructions of https://ch.mathworks.com/help/matlab/matlab-engine-for-python.html <br />
you first have to find your MatLab root folder by opening MatLab and running the command matlabroot. <br />
This should give you the root folder for Matlab. <br />

Then you open your terminal (if you are using Windows you can do that by pressing Windows + R, 
then type cmd and press Enter.) <br />
In the terminal you run following code: <br />

cd matlabroot\extern\engines\python <br />
Make sure to replace matlabroot with the Path you just found. Then you run <br />

python3 setup.py install <br />
To install the MatLab Python library. <br />

2. Using the MatLab Library <br />

Following the instructions of <br />
 https://ch.mathworks.com/help/matlab/matlab_external/call-matlab-functions-from-python.html You can then <br />

import matlab.engine <br />

==========================================
Follow the tutorial for running the code
### License

Copyright © 2021 Sk. Arif Ahmed

The content of this repository is bound by the following licenses:

- The documents and data are licensed under the MIT license.
