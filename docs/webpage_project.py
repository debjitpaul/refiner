from flask import Flask
import random
import socket
import argparse
import numpy as np
from flask import render_template
from flask import request, jsonify
from services.model_service import ModelService
from services.data_service import DataService, oracle_hint

parser = argparse.ArgumentParser(description='webpage actor critic')
# data specific
parser.add_argument('--random-data', default=False, type=bool,
                    help='true is data selection random')
parser.add_argument('--data-number', default=10, type=int,
                    help='number of data by file to display in the select bar')
# model specific
parser.add_argument('--actor-path',
                    type=str, help='path of the actor model')
parser.add_argument('--critic-path',
                    type=str, help='path of the critic model')
# other
parser.add_argument('--verbose', default=True, type=bool,
                    help='print input and output of models to help debugging')
parser.add_argument('--run-EPFL-cluster', default=False, action="store_true",
                    help='True if run on EPFL cluster else use the host')

app = Flask(__name__)


# home route
@app.route('/')
def home():
    """load html page home
    """
    return render_template('home.html')


# data selection POST methods
@app.route('/data_selection', methods=['POST'])
def data_selection():
    """load data in the select bar

    Returns:
        math_pb and label: math problem do dispay and there labels (index;file)
    """
    math_pb = []
    label = []
    # train data
    if request.json["active_train"]:
        # random data
        if args.random_data:
            all_index = [*range(data_service_train.n)]
            ind_train = random.sample(all_index,
                                      args.data_number).astype("int64")
        else:
            ind_train = np.linspace(0, data_service_train.n-1,
                                    args.data_number).astype("int64")
        math_pb = math_pb + data_service_train.math_pb[ind_train].tolist()
        label = label + data_service_train.label[ind_train].tolist()
    # test data
    if request.json["active_test"]:
        # random data
        if args.random_data:
            all_index = [*range(data_service_test.n)]
            ind_test = random.sample(all_index, args.data_number)
        else:
            ind_test = np.linspace(0, data_service_test.n-1,
                                   args.data_number).astype("int64")
        math_pb = math_pb + data_service_test.math_pb[ind_test].tolist()
        label = label + data_service_test.label[ind_test].tolist()
    return jsonify({"math_pb": math_pb, "label": label})


# display the selected problem POST method
@app.route('/display_data', methods=['POST'])
def display_data():
    """get the selected data to display on the page

    Returns:
        problem selected: math problem selected to display
    """
    # get the label
    label = request.json["display_data"].split(";")
    # get index
    index = int(label[0])
    # get filename
    filename = label[1]
    # get problem train data
    if filename == "train.csv":
        problem_select = data_service_train.get_math_pb(index)
    # get problem test data
    elif filename == "test.csv":
        problem_select = data_service_test.get_math_pb(index)
    # get problem other file
    else:
        data_service = DataService(file=r"static/data/" + filename)
        problem_select = data_service.get_math_pb(index)
    return jsonify({"problem_select": problem_select})


# perform actor model fisrt turn POST method
@app.route('/actor_first_turn', methods=['POST'])
def actor_first_turn():
    """Actor model first turn on the selected data

    Returns:
        output: linear equation generate by the model (split by operations)
    """
    # label of the selected data index;file
    label = request.json["select_data"].split(";")
    # file of the selected data
    filename = label[1]
    data_service = DataService(file="static/data/" + filename)
    # index of the selected data
    index = int(label[0])
    # load selected data as global variable
    global data_select
    data_select = data_service.get_item(index)
    # get the math problem selected
    problem = data_service.get_math_pb(index)
    # actor model first turn perform
    first_turn_answer = model.forward_actor_model(input_str=problem, turn=1)
    # split by operation and remove the last EOS
    first_turn_answer = first_turn_answer.split("|")[:-1]
    return jsonify({"output": first_turn_answer})


# perfrom critic model POST method
@app.route('/call_critic', methods=['POST'])
def call_critic():
    """perform critic to generate hint :
    two type of critic automatic (critic model) and oracle (the true critic
    based on the true linear equation)

    Returns:
        hint: the generate hint, true_linear_formula: oracle critic case
    """
    critic_mode = request.json["critic_mode"]
    # automatic critic
    if critic_mode == "automatic":
        # perform critic
        hint = model.forward_critic_model()
        return jsonify({"output": hint})
    # oracle critic
    if critic_mode == "oracle":
        # get generate linear formula by the first turn actor model
        generate_linear_formula = model.history[-1][1][:-6]
        # get the true one
        linear_formula = data_select["linear_equation"]
        # generate true hint
        hint = oracle_hint(generate_linear_formula, linear_formula)
        # append hint to history
        model.history[-1].append(hint)
        return jsonify({"output": hint, "true_linear_formula": linear_formula})


# perform actor model second turn POST method
@app.route('/actor_second_turn', methods=['POST'])
def actor_second_turn():
    """Actor model second turn on the selected data and with the selected hint

    Returns:
        output : linear formula generate by the model
    """
    critic_mode = request.json["critic_mode"]
    # manual hint
    if critic_mode == "manual":
        # get the hint
        hint = request.json["hint_input"]
        # append hint to history
        model.history[-1].append(hint)
    # oracle and automatic hint
    else:
        # get hint from history
        hint = model.history[-1][-1]
    # actor model second turn perform
    second_turn_answer = model.forward_actor_model(input_str=hint, turn=2)
    # split by operation and remove the last EOS
    second_turn_answer = second_turn_answer.split("|")[:-1]
    return jsonify({"output": second_turn_answer})


if __name__ == '__main__':

    args = parser.parse_args()

    # load train and test data service
    data_service_train = DataService(file=r"static/data/train.csv")
    data_service_test = DataService(file=r"static/data/dev.csv")

    # load actor and critic model
    print("Loading actor and critic model...")
    model = ModelService(args.actor_path, args.critic_path,
                         verbose=args.verbose)
    print("Model load well")

    # run the webpage on EPFL serveur
    if args.run_EPFL_cluster:
        app.run(port=8080, host="10.90.38.4", debug=True)
    # run the webpage on the host IP
    else:
        hostname = socket.gethostname()
        # getting the IP address using socket.gethostbyname() method
        # ip_address = socket.gethostbyname(hostname)
        ip_address = "localhost"
        app.run(port=8080, host=ip_address, debug=True)
