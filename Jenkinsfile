pipeline {
  stage('Build') {
    steps {
        parallel (
            "Windows" : {
                echo 'done'
            },
            "Linux" : {
                echo 'done'
            }
       )
    }
  }
}  
