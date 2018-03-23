pipeline {
    agent none
    stages {
        stage('Build container') {
            parallel {
                stage('Set up GCC7 container') {
                    agent {
                        dockerfile {
                            additionalBuildArgs  '--build-arg gccvers=7' 
                        }
                    }
                    environment {
                        HOME = "${WORKSPACE}"
                    }
                    steps {
                        sh "which python ; python --version ; gcc-7 --version"  
                    }
                    post {
                        success {
                            echo "Built "
                        }
                    }
                }
            }
        }
    }
}
