pipeline {
    agent none
    stages {
        stage('Run Tests') {
            parallel {
                stage('Build and test with gcc7') {
                    agent {
                        dockerfile {
                            additionalBuildArgs  '--build-arg gccvers=-7'
                        }
                        steps {
                            echo "steps"
                        }
                        post {
                            always {
                                echo "post"
                            }
                        }
                    }
                }
            }
        }
    }
}
