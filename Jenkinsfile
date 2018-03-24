pipeline {
    agent { label 'dockerhost' }
    stages {
        stage('Build container') {
            parallel {
                stage('Build GCC7 container') { steps { buildImage('7', 'yask') } }
            }
        }
    }
}               
                
def buildImage (gccvers, DEVITO_BACKEND) {
    script {
        def customImage = docker.build("opesci/devito-jenkins:gcc7-${env.BUILD_ID}", "-f Dockerfile.jenkins --build-arg gccvers=${gccvers} --build-arg DEVITO_BACKEND=${DEVITO_BACKEND} .")
        customImage.inside {
            sh "which python ; python --version ; gcc --version"
        }
        customImage.push()
        customImage.push('latest')
    }
}
