// Jquery request for:
//    I) ask for answer validity
//    II) critic, hint generation (manual, automatic, oracle)


// I) ask for answer validity
// action on the radio for answer validity
$('#ask_answer').change(function(e){
    e.preventDefault();
    reload_answer_validity()
});
    
function reload_answer_validity(){
    var selected_value = $("input[name='options-outlined']:checked").val();
    var answer_validity = $("#answer_validity");
    var content = ""
    // good answer
    if (selected_value == "good") {
        content = "<strong> Great, problem solved then! </strong>";
    // wrong answer ask for critic
    } else if(selected_value == "wrong"){
        content += "<strong> Do you want to be the critic or rely on the oracle critic ? </strong> &ensp;&ensp;&ensp;&ensp; ";
        content += "<div class='form-check form-check-inline'>";
        content += "<input class='form-check-input' type='radio' name='critic_type' id='manual_critic' value='manual'>";
        content += "<label class='form-check-label' for='critic_type'> Manual critic </label></div>";
        content += "<div class='form-check form-check-inline'>";
        // content += "<input class='form-check-input' type='radio' name='critic_type' id='automatic_critic' value='automatic'>";
        // content += "<label class='form-check-label' for='critic_type'> Automatic critic </label></div>";
        content += "<div class='form-check form-check-inline'>";
        content += "<input class='form-check-input' type='radio' name='critic_type' id='oracle_critic' value='oracle'>";
        content += "<label class='form-check-label' for='critic_type'> Oracle critic </label></div>";
    }
    $("#critic").html("")
    $("#active_critic").html("")
    $("#critic_response").html("")
    answer_validity.html(content);
}
  

// II) critic, hint generation (manual, automatic, oracle)
// action on radio for critic selection
$('#answer_validity').change(function(e){
    e.preventDefault();
    generate_critic()
});
  
// generate hint
// if manual -> ask for the hint
// if automatic -> call critic model on the server
// if oracle -> genrate oracle hint on the server
function generate_critic(){
    var selected_value = $("input[name='critic_type']:checked").val();
    $("#critic_response").html("")
    var critic = $("#critic");
    var active_critic = $("#active_critic");
    var content = "";
    active_critic.html("<button class='btn btn-primary' id='action_resolve'>Refine</button>")
    // manual
    if (selected_value == "manual") {
        content += "<form id='hint'>";
        content += "<label for='hint_input' class='form-label'> <strong>Give a hint to the model</strong></label>";
        content += "<input class='form-control' id='hint_input' placeholder='Sample hint: the operator in the position x is incorrect.' autocomplete='off'></form>";
    }
    // automatic
    else if(selected_value == "automatic"){
      // perform automatic critic on the server
      axios.post('/call_critic', {"critic_mode": selected_value}).then(function (response) {
        content += "The Critic Hint: <strong> " + response.data["output"] + "</strong> </br>";
        critic.html(content);
      })
      .catch(function (error) {
        console.log(error);
      });
    }
    // oracle 
    else if(selected_value == "oracle"){
      // perform oracle critic on the server
      axios.post('/call_critic', {"critic_mode": selected_value}).then(function (response) {
        content += "Based on the true equation <strong>"  + response.data["true_linear_formula"] +"</strong> </br>"
        content += "The oracle hint is : <strong> " + response.data["output"] + "</strong> </br>";
        critic.html(content);
      })
      .catch(function (error) {
        console.log(error);
      });
    }
    critic.html(content);
}  