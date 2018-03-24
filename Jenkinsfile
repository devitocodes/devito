pipeline {
    agent none
    stages {
        stage('Build container') {
            parallel {
                stage('Build GCC7 container') {
                    agent { label 'dockerhost' }
                    steps {
                        script {
                            buildImage(gccvers=7, DEVITO_BACKEND=yask)
                        }    
                    }
                }
            }
        }
    }
}

void buildImage (gccvers, DEVITO_BACKEND) {
    script {
    def customImage = docker.build("opesci/devito-jenkins:gcc7-${env.BUILD_ID}", "-f Dockerfile.jenkins --build-arg gccvers=${gccvers} --build-arg DEVITO_BACKEND=${DEVITO_BACKEND} .")
    customImage.inside {
        sh "which python ; python --version ; gcc-7 --version"
    }
    customImage.push()
    customImage.push('latest')
}

