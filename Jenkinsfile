pipeline {
    agent { label 'dockerhost' }
    stages {
        stage('Build container') {
            parallel {
                stage('Build gcc-4.9 container') { steps { buildImage('4.9') } }
                stage('Build gcc-4.9 OpenMP container') { steps { buildImage('4.9', null, '1', '2') } }
                stage('Build gcc-5 container') { steps { buildImage('5') } }
                stage('Build gcc-7 container') { steps { buildImage('7', 'yask') } }
            }
        }
    }
}               
                
def buildImage (def gccvers, def DEVITO_BACKEND=null, def DEVITO_OPENMP=null, def OMP_NUM_THREADS=null) {
    script {
        BACKEND_ARG = (DEVITO_BACKEND!=null && DEVITO_BACKEND.length()>0) ? "--build-arg DEVITO_BACKEND=${DEVITO_BACKEND}" : ''
        def customImage = docker.build("opesci/devito-jenkins:gcc${gccvers}-${env.BUILD_ID}", "--no-cache -f Dockerfile.jenkins --build-arg gccvers=${gccvers} ${BACKEND_ARG} .")
        customImage.inside {
            sh "which python ; python --version ; gcc --version"
        }
        customImage.push()
        customImage.push('latest')
    }
}
