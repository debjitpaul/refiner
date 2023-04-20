import os
import sys 
import json
import random

from tqdm import tqdm


class Negative_Examples: 
    def __init__(self,
                data_file):
        self.data_file = data_file

    def create(self, data_file):
        """
        Create Negative Examples
        """
        data = []
        for i, pair in tqdm(enumerate(json.load(open(data_file, 'r')))): 
            text_a = pair['Body'] + pair['Question']
            label = str(pair['Linear_Formula'])
            label = label.replace("'","")
            perturbed_numbers, perturbed_operations, perturbed_label_removed, perturbed_label_add = self.perturb(label)

            print("************")
            print(label)
            print(perturbed_numbers)
            print(perturbed_operations)
            print(perturbed_label_removed)
            print(perturbed_label_add)

            feedback_numbers = self._critique_function(perturbed_numbers, label)
            feedback_operations = self._critique_function(perturbed_operations, label)
            feedback_add = self._critique_function(perturbed_label_removed, label)
            feedback_remove = self._critique_function(perturbed_label_add, label)

            #print(feedback_add, feedback_remove)
            
            print(feedback_numbers, feedback_operations, feedback_remove, feedback_add)
            
            instance_1 = self.get_new_instances(pair, perturbed_numbers, feedback_numbers[0])
            #if " <hint>  No" not in feedback_numbers[0]:
            data.append(instance_1)
            print(instance_1)
            instance_2 = self.get_new_instances(pair, perturbed_operations, feedback_operations[0])
            #if " <hint>  No" not in feedback_operations[0]:
            data.append(instance_2)
            instance_3 = self.get_new_instances(pair, perturbed_label_removed, feedback_add[0])
            #if " <hint>  No" not in feedback_add[0]:
            data.append(instance_3)
            instance_4 = self.get_new_instances(pair, perturbed_label_add, feedback_remove[0])
            #if " <hint>  No" not in feedback_remove[0]:
            data.append(instance_4)
            
        with open('refiner/data/finetune_feedback_data.json', 'w') as json_file:
            json.dump(data, json_file, indent=0, sort_keys=True)
        print('Successfully appended to the JSON file')
            

            
    def get_new_instances(self, pair, perturbed_label, feedbacks):
        instance = {}
        instance['Body'] = pair['Body'] + ' ' + pair['Question']
        instance['Question'] = " Previous Answer: " + str(perturbed_label) +' ' + str(feedbacks)
        instance['Linear_Formula'] = pair['Linear_Formula']
        return instance    

    
    def occurance_operations(self, operation_list, equation): 
        operations = []
        for i in range(len(operation_list)):
            if operation_list[i] in equation:
                operations.append(operation_list[i])
        return operations

    def occurance_values(self,number_list, equation): 
        numbers = []
        for i in range(len(number_list)):
            if number_list[i] in equation:
                numbers.append(number_list[i])
        return numbers

    def remove_operations(self, equation):
        new_equation = ""
        for i in equation.split(' '):
            if i !="|": new_equation += i+ ' '
            else:
                new_equation += i 
                break            
        return new_equation + ' EOS'

    def operator_perturbed(self, label, operations): 
        operation_list = ['add', 'subtract', 'divide','multiply']
        dummy_list = operation_list[:]
        perturbed_label = label
        for i in label.split(' '): 
            if i in operation_list: 
                #print(dummy_list)
                dummy_list.remove(str(i))
                #print(i, dummy_list)
                perturbed_label = label.replace(i, random.choice(dummy_list))
            dummy_list = operation_list[:]

        return perturbed_label 
    
    def add_operations(self, equation, operation_dict, numbers, operations):
        correct_value = random.choice(numbers)
        correct_operations = random.choice(operations)
        new_equation = equation.split('| ')[0] + '| #'+str(1) +": " + correct_operations +' ( '+ '#'+str(0)+', '+correct_value+' )' +' | EOS'        
   
        return new_equation

    def number_perturbed(self, label, numbers):
        number_list = ['number0', 'number1', 'number2'] #'number3', 'number4', 'number5', 'number6', '#0', '#1', '#2', '#3']
        comma_number_list = ['number0,', 'number1,' ,'number2,', 'number3,']
        perturbed_label = label
        dummy_list = comma_number_list[:]
        for i in label.split(' '): 
            if i in number_list: 
                try:
                    perturbed_label = perturbed_label.replace(i, random.choice(number_list))
                except IndexError:
                    perturbed_label = perturbed_label.replace(i, random.choice(number_list))
            if i in comma_number_list:
                dummy_list.remove(str(i))
                perturbed_label = perturbed_label.replace(i, random.choice(dummy_list))
            dummy_list = comma_number_list[:]
        return perturbed_label 

    def count_operations(self, operation_list, label):
        count = 0 
        for i in label.split(' '):
            if i in operation_list: 
                count+=1
        return count

    def perturb(self, label):
        perturbed_label  = label
        perturbed_label_removed = label
        operation_list = ['add', 'subtract', 'divide','multiply']
        number_list = ['number0', 'number1', 'number2', 'number3', 'number4', 'number5', 'number6', '#0', '#1', '#2', '#3']
        operation_dict = {'+': 'add', '-':'subtract', '/': 'divide', '*': 'multiply'}
        
        operations = self.occurance_operations(operation_list, label)
        numbers = self.occurance_values(number_list, label)
        perturbed_label_numbers = self.number_perturbed(label, numbers)
        perturbed_label_operations = self.operator_perturbed(label, operations)
        perturbed_label_add = label
        count_opt = self.count_operations(operation_list, label)
        #print(operations,len(operations))
        #print(numbers, len(numbers))
        
        if count_opt>=2:
             perturbed_label_removed = self.remove_operations(label)
        
        if count_opt==1:
            perturbed_label_add = self.add_operations(label, operation_dict, numbers, operations)

        return perturbed_label_numbers, perturbed_label_operations, perturbed_label_removed, perturbed_label_add
    
    def _critique_function(self, generated_explanation, gold_explanation):
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
                hint = hint + self.gen_hint(list_eq1, difference_position)
            else: 
                difference_position = [pos for pos in range(len(list_eq2)) if list_eq2[pos] != list_eq1[pos]]
                difference_position.extend([pos for pos in range(len(list_eq2), len(list_eq1))])
                hint = hint + self.gen_hint(list_eq1, difference_position)
        
        if hint=="":
            hint = [" <hint>  No" + " | EOH "]
        else:
            hint = [hint + " | EOH "]

        hints.extend(hint)

        return hints

    def gen_hint(self, equation, difference_position):
        hint = ""
        regret = 0 
        operation_list = ['add', 'subtract', 'divide','multiply']
        number_list = ['number0', 'number1', 'number2', 'number3', 'number4', 'number5', 'number6', '#0', '#1', 'number1,','number2,', 'number0,', 'number3,', '#0,', '#1,']
        print(difference_position)
        for index in difference_position: 
            
            if equation[index] in operation_list:
                if index <7:  
                    hint = hint + "the operator in #"+ str(0)+ " is incorrect. "
                    regret += 0
                elif index >=7 and index <14:  
                    hint = hint + "the operator in #"+ str(1)+ " is incorrect. "
                    regret += 0
                elif index >=14 and index <21:  
                    hint = hint + "the operator in #"+ str(2)+ " is incorrect. "
                    regret += 0
                else: 
                    hint = hint + "the operator in #"+ str(3)+ " is incorrect. "
                    regret += 0
            elif equation[index] in number_list:
                if index <7:
                    if index==3:   
                        hint = hint + "the first number in #"+ str(0)+ " is incorrect. "
                    elif index==4:
                        hint = hint + "the second number in #"+ str(0)+ " is incorrect. "
                    regret +=0
                elif index >=7 and index <14:
                    if index==10:  
                        hint = hint + "the first number in #"+ str(1)+ " is incorrect. "
                    else:
                        hint = hint + "the second number in #"+ str(1)+ " is incorrect. "
                    regret +=0
                elif index >=14 and index <21:  
                    if index==17: 
                        hint = hint + "the first number in #"+ str(2)+ " is incorrect. "
                        
                    else:
                        hint = hint + "the second number in #"+ str(2)+ " is incorrect. "
                    regret += 0
                else:
                    if index==25:  
                        hint = hint + "the first number in #"+ str(3)+ " is incorrect. "
                    else:
                        hint = hint + "the second number in #"+ str(3)+ " is incorrect. "
                    regret += 0

        return hint
    
data_file = "/root/refiner/data/critic_mwp/extra_data.json"
my_new_instance = Negative_Examples(data_file)
my_new_instance.create(data_file)

    



    
