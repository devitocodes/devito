pipeline {
    agent none
    stages {
        stage('Run Tests') {
            parallel {
                stage('Test On Linux 1') {
                    agent {
                        dockerfile true
                    }
                    steps {
                        echo "Steps"
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
