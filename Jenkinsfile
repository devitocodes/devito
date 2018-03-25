pipeline {
    agent { label 'dockerhost' }
    stages {
        stage('Build and Test') {
            parallel {
                stage('Build and test gcc-4.9 container') { steps { buildAndTest('4.9') } }
                stage('Build and test gcc-4.9 OpenMP container') { steps { buildAndTest('4.9', null, '1', '2') } }
                stage('Build and test gcc-5 container') { steps { buildAndTest('5') } }
                stage('Build and test gcc-7 container') { steps { buildAndTest('7', 'yask') } }
            }
        }
    }
}               
                
def buildAndTest (def gccvers, def DEVITO_BACKEND=null, def DEVITO_OPENMP=null, def OMP_NUM_THREADS=null) {
    script {
        BACKEND_ARG = (DEVITO_BACKEND!=null && DEVITO_BACKEND.length()>0) ? "--build-arg DEVITO_BACKEND=${DEVITO_BACKEND}" : ''
        BACKEND_SUFFIX = (DEVITO_BACKEND!=null && DEVITO_BACKEND.length()>0) ? "-${DEVITO_BACKEND}" : ''
        def customImage = docker.build("opesci/devito-jenkins:gcc${gccvers}${BACKEND_SUFFIX}-${env.BUILD_ID}", "--no-cache -f Dockerfile.jenkins --build-arg gccvers=${gccvers} ${BACKEND_ARG} .")
        customImage.inside {
            sh "flake8 --builtins=ArgumentError ."
            sh "py.test -vs --cov devito tests/"
        }
        customImage.push()
        customImage.push('latest')
    }
}
