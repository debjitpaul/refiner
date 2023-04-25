// Jquery request for:
//    I) actor model first turn
//    II) actor model second turn (with hint)


// I) actor model first turn
// action on the solve button
// we want:
// 1) to send the data to the server
// 2) get the response of the server (the response of the actor model fisrt turn)
// 3) ask if the response is correct
$("#action_select_data").on('click', function(e) {
    e.preventDefault();
    // get the select data label
    var select_data = $("#select_data").val();
    if (select_data != -1){
        // remove section
        $("#answer_validity").html("")
        $("#critic").html("")
        $("#active_critic").html("")
        $("#critic_response").html("")
        $("#response_model").html("")
        $("#ask_answer").html("")
        // send to server
        axios.post('/actor_first_turn', {"select_data": select_data})
          .then(function (response) {
            console.log(response);
            // disp answer of the model
            var div_response_model = $("#response_model");
            var content = "<strong>Generator Output</strong><br>"
            for (let [i, opt] of Object.entries(response.data["output"])){
              // if (i>0){
              //   content += "&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp; <strong>" + opt + "</strong> <br>"
              // }
              // else{
              content += opt + "<br>"
              // }
            }
            div_response_model.html(content);
            // ask for true or false aswer
            var answer_ask = $("#ask_answer");
            var ask = "<strong>Is the answer correct?</strong> &ensp;&ensp;&ensp;&ensp; ";
            // correct
            ask += "<input type='radio' class='btn-check' name='options-outlined' id='success-outlined' autocomplete='off' value='good' >";
            ask += "<label class='btn btn-outline-success' for='success-outlined'>Correct</label>  &ensp;&ensp;&ensp;&ensp;";
            // wrong
            ask += "<input type='radio' class='btn-check' name='options-outlined' id='danger-outlined' autocomplete='off' value='wrong' >"
            ask += "<label class='btn btn-outline-danger' for='danger-outlined'>Wrong</label>"
            answer_ask.html(ask);
          })
          .catch(function (error) {
            console.log(error);
          });
    }
  });


// II) actor model second turn
// action on the resolve button
// we want:
// 1) to send the data to the server
// 2) get the response of the server (the response of the actor model second turn)
$("#active_critic").on('click', function(e) {
    e.preventDefault();
    // get critic type
    var critic_mode = $("input[name='critic_type']:checked").val();
    if (critic_mode == "manual") {
      var hint_input = $("#hint_input").val();
    }
    else{
      var hint_input = "nan"
    }
    axios.post('/actor_second_turn', {"critic_mode": critic_mode, "hint_input": hint_input}).then(function (response) {
      // disp response of the model
      var critic_response = $("#critic_response");
      var content = "<strong>Refined Generator Output</strong><br>"
      for (let [i, opt] of Object.entries(response.data["output"])){
        // if (i>0){
        //   content += "&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <strong>" + opt + "</strong> <br>"
        // }
        // else{
        content += opt + "<br>"
        // }
      }
      critic_response.html(content);
    })
    .catch(function (error) {
    console.log(error);
    });
  });