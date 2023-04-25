// Jquery request for:
//    I) dataset selection
//    II) display select data


// I) dataset selection
// action on checked train datset
$("#active_train").on('change', function(e) {
    e.preventDefault();
    reload_model_selection();
  });
  
  // action on checked test dataset
  $("#active_test").on('change', function(e) {
    e.preventDefault();
    reload_model_selection();
  });
  
  // load data from the checked dataset
  function reload_model_selection(){
    var active_train = $("#active_train").is(":checked");
    var active_test = $("#active_test").is(":checked");
    axios.post('/data_selection', {"active_train": active_train, "active_test": active_test}).then(function (response) {
      var select_data = $("#select_data");
      var options =  "<option value='-1' hidden>Choose a Math problem</option>";
      for (let [i, opt] of Object.entries(response.data["math_pb"])){
        options += "<option value='" + response.data["label"][i] + "'>" +  opt + "</option>";
      }
      select_data.html(options);
      })
    .catch(function (error) {
      console.log(error);
    });
}
  

// II) display selected data
// action select data on the select bar
// 1) send the label of the select data to the server
// 2) display the select data
$("#select_data").on('change', function(e){
    display_data = $("#display_data");
    var select_data = $("#select_data").val();
    if (select_data != -1){
        $("#response_model").html("")
        $("#ask_answer").html("")
        $("#answer_validity").html("")
        $("#critic").html("")
        $("#active_critic").html("")
        $("#critic_response").html("")
        axios.post('/display_data', {"display_data": select_data})
        .then(function (response) {
            console.log(response);
            var content = "<span><strong>Chosen Problem</strong></span><br>" + response.data["problem_select"]
            display_data.html(content)
        })
    }
});