pipeline {
  agent {
    docker { image 'python:3.5.1' }
  }
  environment {
    PATH = "/usr/local/bin:/usr/bin:/bin"
  }
  stages {
    stage('1') {
      steps {
        sh 'true'
      }
    }
  }
} 
