pipeline {
    agent none
    stages {
        stage('Build container') {
            parallel {
                stage('Build GCC7 container') {
                    agent { label 'dockerhost' }
                    steps {
                        script {
                            def customImage = docker.build("opesci/devito-jenkins:gcc7-${env.BUILD_ID}", "--build-arg gccvers=7 --build-arg DEVITO_BACKEND=yask .")
                            customImage.inside {
                                sh "which python ; python --version ; gcc-7 --version"
                            }
                            customImage.push()
                            customImage.push('latest')
                        }    
                    }
                    post {
                        success {
                            echo "Built devito-gcc7:${env.BUILD_ID}"
                        }
                    }
                }
            }
        }
        stage ('Run something in the GCC7 container') {
            agent { docker "opesci/devito-jenkins:gcc7-${env.BUILD_ID}" }
            steps {
                sh "which python ; python --version ; gcc-7 --version"
            }
        }
    }
}
