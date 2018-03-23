pipeline {
    agent none
    stages {
        stage('Run Tests') {
            parallel {
                stage('Test On Linux 1') {
                    agent {
                        dockerfile { additionalBuildArgs  '--build-arg gccvers=7' }
                    }
                    steps {
                        sh "wget http://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ~/miniconda.sh"
                        sh "bash ~/miniconda.sh -b -p $HOME/miniconda"
                        
                    }
                    post {
                        always {
                            echo "Post"
                        }
                    }
                }
            }
        }
    }
}
