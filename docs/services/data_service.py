import pandas as pd
import numpy as np


class DataService():
    """Service to manage data
    """
    def __init__(self, file):
        """init from file
        """
        self.file = file
        # load file as pd dataframe
        self.data = pd.read_csv(file)
        # get features
        self.numbers = self.data["Numbers"]
        self.body = self.data["Body"]
        self.question = self.data["Ques"]
        self.answer = self.data["Answer"]
        # get number of data
        self.n = self.numbers.shape[0]

        # manage dataset
        self.create_math_pb()
        self.create_label()
        self.create_linear_equation()
        # drop data with incorrect linear formula
        index_names = self.data[self.data['linear_equation'] == 'NAN'].index
        self.data.drop(index_names, inplace=True)

    def create_math_pb(self):
        """create math problem = Body + Question
            and clean the syntaxe
        """
        clean_body = self.body.values
        clean_question = self.question.values
        # manage no question data
        for i in range(len(self.numbers)):
            if not isinstance(clean_question[i], str):
                clean_question[i] = ""
        # clean the syntax
        clean_body = np.array([self.clean_sentence(i) for i in clean_body])
        clean_question = np.array([self.clean_sentence(i, True) for i in clean_question])
        self.math_pb = []
        # create math problem
        for body, quest in zip(clean_body, clean_question):
            self.math_pb.append(body + " " + quest)
        self.math_pb = np.array(self.math_pb)

    def create_label(self):
        """create label for data : index:filename
        """
        file_name = self.file.split("/")[-1]
        self.data["label"] = self.data.index.astype('str') + ";" + file_name
        self.label = self.data["label"].values

    def create_linear_equation(self):
        """create linear equation for all data
        """
        linear_equation = []
        for equation in self.data["Equation"]:
            linear_equation.append(self.tree2linear(equation)[:-3])
        self.data["linear_equation"] = linear_equation

    def get_item(self, index):
        """get item

        Args:
            index (int): index

        Returns:
            dict: data selected
        """
        return self.data.iloc[index].to_dict()

    def get_math_pb(self, index):
        """get math problem

        Args:
            index (int): index

        Returns:
            str: selected math problem
        """
        return self.math_pb[index]

    def tree2linear(self, equation):
        """
        - + number1 number2 number3 --> add(number1, number2) | subtract(#0, number3)
        - number1 + number2 number3 --> add(number2, number3) | subtract(#0, number1)
        """
        number_list = ['number0', 'number1', 'number2', 'number3', 'number4', 'number5', 'number6']
        operation_dict = {'+': 'add', '-': 'substract', '/': 'divide', '*': 'multiply'}
        operation_list = ['*', '+', '-', '/']

        # equation = "- number1 + number2 number3"
        # equation = "* - 1.0 * + 1.0 * number0 0.01 - 1.0 * number0 0.01 100.0"
        operations, numbers = self._occurance(operation_list, number_list, equation)
        o_index, _, c_index = self.get_index(operation_list, number_list, equation)
        list_equation = equation.split(' ')
        if len(operations) == 1:
            i = 0
            if c_index == []:
                linear_equation = '#'+str(0) + ": " + operation_dict[operations[0]] + ' ( '+numbers[i] + ', '+numbers[i+1]+' )' + ' | '
            else:
                linear_equation = '#'+str(0) + ": " + operation_dict[operations[0]] + ' ( '+list_equation[i+1] + ', '+list_equation[i+2]+' )' + ' | '

        elif len(operations) == 2:
            # if the pattern is operation2 operation1 numberX numberY numberZ
            operations = operations[::-1]
            if 0 in o_index and 1 in o_index:
                for i in range(len(operations)):
                    if i == 0:
                        linear_equation = '#'+str(i) + ": " + operation_dict[operations[i]] + ' ( '+list_equation[i+2] + ', '+list_equation[i+3]+' )' + ' | '
                    else:
                        linear_equation = linear_equation + '#'+str(i) + ": " + operation_dict[operations[i]]+' ( ' + '#'+str(i-1) + ', '+list_equation[i+3]+' )' + ' | '
            else:
                # if the pattern is operation2 numberZ operation1 numberX numberY
                start_index = len(operations)
                for i in range(len(operations)):
                    if i == 0:
                        linear_equation = '#'+str(i) + ": " + operation_dict[operations[i]] + ' ( '+list_equation[start_index+i+1] + ', '+list_equation[start_index+i+2] + ' )' + ' | '
                    else:
                        linear_equation = linear_equation + '#'+str(i) + ": " + operation_dict[operations[i]] + ' ( '+list_equation[start_index-i] + ', #'+str(i-1) + ' )' + ' | '

        elif len(operations) > 2:
            operations = operations[::-1]
            start_index = len(operations)
            # check if the o_index is incrementing one at a time 0,1,2,3,..., len(operations)
            if not o_index[0] and all(y-x == 1 for x, y in zip(o_index, o_index[1:])):
                for i in range(len(operations)):
                    if i == 0:
                        linear_equation = '#'+str(i) + ": " + operation_dict[operations[i]] + ' ( ' + list_equation[start_index+i] + ', ' + list_equation[start_index+i+1] + ' )' + ' | '
                    else:
                        linear_equation = linear_equation + '#'+str(i) + ": " + operation_dict[operations[i]] + ' ( ' + '#'+str(i-1) + ', ' + list_equation[start_index+i+1] + ' )' + ' | '

            else:
                linear_equation = "NAN"
        else:
            linear_equation = "NAN"

        return linear_equation

    @staticmethod
    def _occurance(operation_list, number_list, equation):
        """for tree2linear get the number and operation of an equation"""
        operations = []
        numbers = []
        list_equation = equation.split(' ')

        for i in range(len(list_equation)):
            if list_equation[i] in operation_list:
                operations.append(list_equation[i])
            elif list_equation[i] in number_list:
                numbers.append(list_equation[i])

        return operations, numbers

    @staticmethod
    def get_index(operation_list, number_list, equation):
        """for tree2linear get index of number and operation of an equation"""
        operation_index = []
        value_index = []
        constant_index = []
        list_equation = equation.split(' ')
        for i in range(len(list_equation)):
            if list_equation[i] in operation_list:
                operation_index.append(i)
            elif list_equation[i] in number_list:
                value_index.append(i)
            else:
                constant_index.append(i)

        return operation_index, value_index, constant_index

    @staticmethod
    def clean_sentence(sentence, question=False):
        """clean syntaxe of math problem"""
        # capitalize first letter
        sentence = sentence.capitalize()
        # add , and manage space
        sentence_split = sentence.split(".")
        if sentence_split[-1] == "":
            del sentence_split[-1]
        sentence_split2 = []
        follow = True
        for i in range(len(sentence_split)):
            if i == len(sentence_split)-1 and not(follow):
                follow = True
            elif i == len(sentence_split)-1:
                sentence_split2.append(sentence_split[i])
            elif sentence_split[i][-1].isdigit() and sentence_split[i+1][0].isdigit():
                sentence_split2.append(sentence_split[i] + " " + sentence_split[i+1])
                follow = False
            elif follow:
                sentence_split2.append(sentence_split[i])
            else:
                follow = True
        new_sentence = ""
        for i, s in enumerate(sentence_split2):
            if s != "":
                if s[0] != " ":
                    s = " " + s
                if s[-1] == " ":
                    s = s[:-1]
                new_sentence = new_sentence + "," + s
        if question:
            if new_sentence != "":
                if new_sentence[-1] != "?":
                    new_sentence = new_sentence[:-1] + "?"
            return new_sentence[2:]
        else:
            if new_sentence[-1] != ",":
                return new_sentence[2:] + "."
            else:
                return new_sentence[2:] + " "


# def oracle_hint(linear_equation_gen, linear_equation_true):
#     """create oracle hint based on the genrate and the true linear equation

#     Args:
#         linear_equation_gen (str): generated linear equation
#         linear_equation_true (str): true linear equation

#     Returns:
#         str: oracle hint
#     """
#     # no error
#     if linear_equation_gen == linear_equation_true:
#         return "No"
#     hint = ""
#     nb_operation_gen = len(linear_equation_gen.split('|'))
#     nb_operation_true = len(linear_equation_true.split('|'))
#     n = nb_operation_true
#     # too many operations
#     if nb_operation_gen > nb_operation_true:
#         hint += "Remove an operation. "
#         n = nb_operation_true
#     # not enougth operations
#     if nb_operation_gen < nb_operation_true:
#         hint += "Add an operation. "
#         n = nb_operation_gen
#     # verify operations
#     for i in range(n):
#         operation_true = linear_equation_true.split('|')[i].split(" ")
#         operation_gen = linear_equation_gen.split('|')[i].split(" ")
#         while "" in operation_true:
#             operation_true.remove("")
#         while "" in operation_gen:
#             operation_gen.remove("")
#         if operation_true != operation_gen:
#             # check operator
#             operator_true = operation_true[1]
#             operator_gen = operation_gen[1]
#             if operator_gen != operator_true:
#                 hint += "The operator in the position " + str(i*7+1) + " is incorrect. "
#             # check number
#             num0_true = operation_true[3][:-1]
#             num1_true = operation_true[4]
#             num0_gen = operation_gen[3][:-1]
#             num1_gen = operation_gen[4]
#             if operator_true in ["add", "multiply"]:
#                 # verify swap number for swapable operation
#                 if (num0_gen != num1_true) | (num0_gen != num1_true):
#                     if num0_true != num0_gen:
#                         hint += "The number in the position " + str(i*7+3) + " is incorrect. "
#                     if num1_true != num1_gen:
#                         hint += "The number in the position " + str(i*7+4) + " is incorrect. "
#             else:
#                 if num0_true != num0_gen:
#                     hint += "The number in the position " + str(i*7+3) + " is incorrect. "
#                 if num1_true != num1_gen:
#                     hint += "The number in the position " + str(i*7+4) + " is incorrect. "
#     if hint == "":
#         hint = "No"
#     return hint

def oracle_hint(generated_explanation, gold_explanation):
        '''
        ------------------------
        Parameter: 
        generated explantion: 
        gold explanation: 
        ------------------------
        Output: 
        Hints
        '''
        hints = []
        hints_ids = []
        regret = 0  
        hint = " <hint> "
        h = ""
        if gold_explanation == generated_explanation:
            hint = hint + " No "
            regret = 0
        else:
            list_eq1 = gold_explanation.split(' ')
            list_eq2 = generated_explanation.split(' ') 
            if gold_explanation.count("|")>generated_explanation.count("|"): 
                hint = hint + " add an operator. " 
                regret += 0
            
            elif generated_explanation.count("|")>gold_explanation.count("|"): 
                hint = hint + " remove an operator. "
                regret += 0
            
            if len(list_eq2)>len(list_eq1):
                difference_position = [pos for pos in range(len(list_eq1)) if list_eq2[pos] != list_eq1[pos]]
                h, regret = gen_hint(list_eq1, difference_position, regret)
            else: 
                difference_position = [pos for pos in range(len(list_eq2)) if list_eq2[pos] != list_eq1[pos]]
                difference_position.extend([pos for pos in range(len(list_eq2), len(list_eq1))])
                h, regret = gen_hint(list_eq1, difference_position, regret)
            hint +=h 
        if hint=="":
            hint = [" <hint> No" + " | EOH "]
        else:
            hint = [hint + " | EOH "]
        hints.extend(hint)
        
        return hint[0]

def gen_hint(equation, difference_position, regret):
        hint = ""
        operation_list = ['add', 'subtract', 'divide','multiply']
        number_list = ['number0', 'number1', 'number2', 'number3', 'number4', 'number5', 'number6', '#0', '#1', 'number1,','number2,', 'number0,', 'number3,', '#0,', '#1,']
        #print(difference_position)
        for index in difference_position: 
            if equation[index] in operation_list:
                if index <7:  
                    hint = hint + " the operator in #"+ str(0)+ " is incorrect. "
                    regret += 0
                elif index >=7 and index <14:  
                    hint = hint + " the operator in #"+ str(1)+ " is incorrect. "
                    regret += 0
                elif index >=14 and index <21:  
                    hint = hint + " the operator in #"+ str(2)+ " is incorrect. "
                    regret += 0
                else: 
                    hint = hint + " the operator in #"+ str(3)+ " is incorrect. "
                    regret += 0
            elif equation[index] in number_list:
                if index <7:
                    if index==3:   
                        hint = hint + " the first number in #"+ str(0)+ " is incorrect. "
                    elif index==4:
                        hint = hint + " the second number in #"+ str(0)+ " is incorrect. "
                    regret +=0
                elif index >=7 and index <14:
                    if index==10:  
                        hint = hint + " the first number in #"+ str(1)+ " is incorrect. "
                    else:
                        hint = hint + " the second number in #"+ str(1)+ " is incorrect. "
                    regret +=0
                elif index >=14 and index <21:  
                    if index==17: 
                        hint = hint + " the first number in #"+ str(2)+ " is incorrect. "
                        
                    else:
                        hint = hint + " the second number in #"+ str(2)+ " is incorrect. "
                    regret += 0
                else:
                    if index==25:  
                        hint = hint + " the first number in #"+ str(3)+ " is incorrect. "
                    else:
                        hint = hint + " the second number in #"+ str(3)+ " is incorrect. "
                    regret += 0
        return hint, regret

if __name__ == '__main__':
    """ main to test the class and fct"""
    data = DataService(file=r"static/data/train.csv")
    linear_equation_gen = "#0: multiply ( number2, number1 ) | #1: divide ( number2, #0 ) | #1: divide ( 2, number2 )"
    linear_equation_true = "#0: add ( number0, number1 ) | #1: add ( #0, number2 )"
    print(oracle_hint(linear_equation_gen, linear_equation_true))
