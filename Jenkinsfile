pipeline {
    agent none
    stages {
        stage('Build container') {
            parallel {
                stage('Set up GCC7 container') {
                    agent {
                        dockerfile {
                            label '${BUILD_TAG}-gcc7'
                            additionalBuildArgs  '--build-arg gccvers=7' 
                        }
                    }
                    environment {
                        HOME = ${WORKSPACE}
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
