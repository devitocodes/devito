pipeline {
    agent none
    stages {
        stage('Build container') {
            parallel {
                stage('Build container') {
                    buildImage(gccvers='7', DEVITO_BACKEND='yask')
                }
            }
        }
    }
}

def buildImage (gccvers, DEVITO_BACKEND) {
    agent { label 'dockerhost' }
    environment { 
        gccvers = gccvers
        DEVITO_BACKEND = DEVITO_BACKEND
    }
    steps {
        script {
            def customImage = docker.build("opesci/devito-jenkins:gcc7-${env.BUILD_ID}", "-f Dockerfile.jenkins --build-arg gccvers=${gccvers} --build-arg DEVITO_BACKEND=${DEVITO_BACKEND} .")
            customImage.inside {
                sh "which python ; python --version ; gcc --version"
            }
            customImage.push()
            customImage.push('latest')
        }
    }
}
