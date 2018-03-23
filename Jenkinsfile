pipeline {
    agent none
    stages {
        stage('Build container') {
            parallel {
                stage('Set up GCC7 container') {
                    agent { label 'dockerhost' }
                    steps {
                        script {
                            def customImage = docker.build("devito-gcc7:${env.BUILD_ID}", "--build-arg gccvers=7 .")
                            customImage.inside {
                                sh "which python ; python --version ; gcc-7 --version"  
                            }
                        }    
                    }
                    post {
                        success {
                            echo "Built devito-gcc7:${env.BUILD_ID} as ${customImage}"
                        }
                    }
                }
            }
        }
    }
}
