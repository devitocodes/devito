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
        BACKEND_ARG = DEVITO_BACKEND ? "--build-arg DEVITO_BACKEND=${DEVITO_BACKEND}" : ''
        BACKEND_SUFFIX = DEVITO_BACKEND ? "-${DEVITO_BACKEND}" : ''
        def customImage = 
            docker.build("opesci/devito-jenkins:gcc${gccvers}${BACKEND_SUFFIX}-${env.BUILD_ID}", 
                         "--no-cache -f Dockerfile.jenkins --build-arg gccvers=${gccvers} ${BACKEND_ARG} .")
        customImage.inside {
            sh "flake8 --builtins=ArgumentError ."
            sh "py.test -vs --cov devito tests/"
            if ( DEVITO_BACKEND!='yask' ) {
                sh "DEVITO_BACKEND=foreign py.test -vs tests/test_operator.py -k TestForeign"
                sh "python examples/seismic/benchmark.py test -P tti -so 4 -a -d 20 20 20 -n 5"
                sh "python examples/seismic/benchmark.py test -P acoustic -a"
                sh "python examples/seismic/acoustic/acoustic_example.py --full"
                sh "python examples/seismic/acoustic/acoustic_example.py --constant --full"
                sh "python examples/seismic/acoustic/gradient_example.py"
                sh "python examples/misc/linalg.py mat-vec mat-mat-sum transpose-mat-vec"
                sh "python examples/seismic/tti/tti_example.py -a"
                sh "python examples/checkpointing/checkpointing_example.py"
                // Test tutorial notebooks for the website using nbval
                sh "py.test -vs --nbval examples/seismic/tutorials"
                sh "py.test -vs --nbval examples/cfd"
            }
            sh "codecov"
        }
        customImage.push()
        customImage.push('gcc${gccvers}${BACKEND_SUFFIX}-latest')
    }
}
