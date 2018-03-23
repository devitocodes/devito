pipeline {
  agent none
  environment {
    PATH = "/usr/local/bin:/usr/bin:/bin"
  }
  stages {
    stage('prep'}{
      parallel {
        stage('xenial-gcc7'){
          agent { 
            label "docker" 
          }
          steps {
            def customImage = docker.build("my-image:${env.BUILD_ID}", "--build-arg gccvers=-7")
            customImage.inside {
              checkout scm
              sh 'ls -l ; gcc --version'
            }
          }
        }
        stage('xenial-gcc8'){
          agent { 
            label "docker" 
          }
          steps {
            def customImage = docker.build("my-image:${env.BUILD_ID}", "--build-arg gccvers=-8")
            customImage.inside {
              checkout scm
              sh 'ls -l ; gcc --version'
            }
          }
        }
      }
    }
  }
}
        
        
