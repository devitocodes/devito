pipeline {
  agent {
    docker { image 'ubuntu' }
  }
  environment {
    PATH = "/usr/local/bin:/usr/bin:/bin"
  }
  stages {
    stage('1') {
      steps {
        sh 'adduser --disabled-password --gecos "" --uid `stat -c "%u" Jenkinsfile` devito'
        sh 'ls -l'
      }
    }
  }
} 
