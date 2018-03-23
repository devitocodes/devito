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
                        sh "which python ; python --version ; gcc-7 --version"
                        
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
