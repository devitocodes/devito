/* Main pipeline is declarative; buildAndTest method embeds some scripted. When it's possible to 
     use the docker.build() method in declarative pipeline, buildAndTest should be rewritten into
     declarative to futureproof this Jenkinsfile. At present, this isn't supported */
       
pipeline {
    // Run on an agent which supports docker
    agent { label 'dockerhost' }
    stages {
        // So we can easily pass container ID to testing, build and test in one stage
        stage('Build and Test') {
            // ... but independent test builds can run in parallel
            parallel {
                // For each combination of parameters required, build and test
                stage('Build and test gcc-4.9 container') { steps { buildAndTest('4.9') } }
                // Note that null needs to be passed as groovy doesn't allow VAR=VAL when calling a method
                stage('Build and test gcc-4.9 OpenMP container') { steps { buildAndTest('4.9', null, '1', '2') } }
                stage('Build and test gcc-5 container') { steps { buildAndTest('5') } }
                stage('Build and test gcc-7 container') { steps { buildAndTest('7', 'yask') } }
            }
        }
    }
}               

// Scripted pipeline method for building and testing a docker container with devito
def buildAndTest (def gccvers, def DEVITO_BACKEND=null, def DEVITO_OPENMP=0, def OMP_NUM_THREADS=1) {
    // Switch into scripted pipeline
    script {
        // If we're specifying a backend, pass it through, and label the container appropriately
        BACKEND_ARG = DEVITO_BACKEND ? "--build-arg DEVITO_BACKEND=${DEVITO_BACKEND}" : ''
        BACKEND_SUFFIX = DEVITO_BACKEND ? "-${DEVITO_BACKEND}" : ''
        OPENMP_SUFFIX = (DEVITO_OPENMP=='1') ? "-openmp" : ''
        /* Now build the container, saving the returned image value to run with testing
           Note that the context directory has to be the final parameter passed */
        def customImage = 
             docker.build("opesci/devito-jenkins:gcc${gccvers}${BACKEND_SUFFIX}${OPENMP_SUFFIX}-${env.BUILD_ID}", 
                         "--no-cache -f Dockerfile.jenkins --build-arg gccvers=${gccvers} ${BACKEND_ARG} .")
        // If the build succeeded, push the container to dockerhub for debugging purposes
        customImage.push()
        // Also push a latest tagged version for easy reference
        customImage.push('gcc${gccvers}${BACKEND_SUFFIX}${OPENMP_SUFFIX}-latest')
        // Using the built container, run tests
        customImage.inside("-e DEVITO_OPENMP=${DEVITO_OPENMP} -e OMP_NUM_THREADS=${OMP_NUM_THREADS}") {
            sh "flake8 --builtins=ArgumentError ."
            sh "py.test -n 2 -vs --cov"
            // Additional seismic operator tests
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
            // Now run coverage
            sh "codecov"
            // Now build documentation
            sh "sphinx-apidoc -f -o docs/ examples/"
            sh "sphinx-apidoc -f -o docs/ devito/ devito/yask/*"
            sh "make -C docs/ html"
        }
    }
}
