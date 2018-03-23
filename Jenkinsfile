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
                        sh "ls /usr/bin"
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
