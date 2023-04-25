# webpageMathPbActorCritic
Webpage for result presentation of Actor Critic method for Math problem answering

Semester project of [Emilien Seiler](mailto:emilien.seiler@epfl.ch), Master in Computational Science and Engineering at EPFL. 
Project in collaboration with Atificial Intelligence Laboratory and Natural Language Processing Laboratory at EPFL.

## Structure

This is the structure of the repository:

- `services`: python class and function for the server
  - `data_service.py`: service for data management on the server
  - `model_service.py`: service for actor and critic model management on the server
- `static`: 
  - `css`: webpage style
    - `my_style.css`: personalized style
  - `data`: dataset
    - `Math-problem-nlp.pdf`: download pdf on the webpage (to change)
    - `dev.csv`: test set
    - `train.csv`: train set
  - `img` : img dispay on the webpage
  - `js` : json file with jQuery request for on click/change action
    - `actor_model.js`: actor model related action
    - `critic_model.js`: critic model related action
    - `data_selection.js`: data selection related action
- `templates`: html template page
  - `home.html`: home page
- `requirement.txt`: python requirement
- `webpage_project.py`: script for run the webpage (see args bellow)


## Data
Data are provided from SVAMP dataset <br> 
Train dataset 3139 math problem 
Test dataset 1000 math problem

## Run
```
python3 webpage_project.py --args <value>
```
#### Arguments: 
Data specific:
- `--random-data`: bool, random data proposed on the webpage (default = False) 
- `--data-number`: int, number of data display in the select bar (default = 10)

Model specific:
- `--actor-path`: str, path of the actor model (default = "static/model/output_reasoning_iterationz) 
- `--critic-path`: str, path of the critic model (default = "static/model/critic")

Other:
- `--verbose`: bool, print input and output of models to help debugging (default = True) 
- `--run-EPFL-cluster`: bool, True if run on EPFL cluster else use the host (default = True)
