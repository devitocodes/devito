pipeline {
    agent { label 'dockerhost' }
    stages {
        stage('Build container') {
            parallel {
                stage('Build gcc-4.9 container') { steps { buildImage('4.9', null) } }
                stage('Build gcc-5 container') { steps { buildImage('5', null) } }
                stage('Build gcc-7 container') { steps { buildImage('7', 'yask') } }
            }
        }
    }
}               
                
def buildImage (def gccvers, def DEVITO_BACKEND) {
    script {
        if (DEVITO_BACKEND!=null && DEVITO_BACKEND.length()>0) {
            BACKEND_ARG="--build-arg DEVITO_BACKEND=${DEVITO_BACKEND}"
        }
        def customImage = docker.build("opesci/devito-jenkins:gcc7-${env.BUILD_ID}", "-f Dockerfile.jenkins --build-arg gccvers=${gccvers} ${BACKEND_ARG} .")
        customImage.inside {
            sh "which python ; python --version ; gcc --version"
        }
        customImage.push()
        customImage.push('latest')
    }
}
