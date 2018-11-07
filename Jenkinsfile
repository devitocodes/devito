
pipeline {
    agent none
    environment {
         LC_ALL='C.UTF-8'
         LANG='C.UTF-8'
         MPLBACKEND='AGG'
         CODECOV_TOKEN='67a18b2a-77f0-4722-9c88-bc1a098473ce'
         DEVITO_LOGGING='INFO'
    }
    stages {
        stage('Testing') {
            parallel {
                // For each combination of parameters required, build and test
                stage('Build and test gcc-4.9 container') {
                     agent { dockerfile { label 'azure-linux-8core'
                                          filename 'Dockerfile.jenkins'
                                          additionalBuildArgs "--build-arg gccvers=4.9" } }
                     environment {
                         HOME="${WORKSPACE}"
                         DEVITO_OPENMP=0
                         PYTHONPATH="${WORKSPACE}/lib/python3.6/site-packages/"
                     }
                     steps {
                         cleanWorkspace()
                         pipInstallDevito()
                         runPipTests()
                     }
                }
                stage('Build and test gcc-4.9 OpenMP container') {
                     agent { dockerfile { label 'azure-linux-8core'
                                          filename 'Dockerfile.jenkins'
                                          additionalBuildArgs "--build-arg gccvers=4.9" } }
                     environment {
                         HOME="${WORKSPACE}"
                         DEVITO_OPENMP=1
                         OMP_NUM_THREADS=2
                     }
                     steps {
                         cleanWorkspace()
                         condaInstallDevito()
                         runCondaTests()
                         runExamples()
                         runCodecov()
                         buildDocs()
                     }
                }
                stage('Build and test gcc-5 container') {
                     agent { dockerfile { label 'azure-linux-8core'
                                          filename 'Dockerfile.jenkins'
                                          additionalBuildArgs "--build-arg gccvers=5" } }
                     environment {
                         HOME="${WORKSPACE}"
                         DEVITO_OPENMP=0
                     }
                     steps {
                         cleanWorkspace()
                         condaInstallDevito()
                         runCondaTests()
                         runExamples()
                         runCodecov()
                         buildDocs()
                     }
                }
                stage('Build and test gcc-7 YASK container') {
                     agent { dockerfile { label 'azure-linux'
                                          filename 'Dockerfile.jenkins'
                                          additionalBuildArgs "--build-arg gccvers=7" } }
                     environment {
                         HOME="${WORKSPACE}"
                         DEVITO_BACKEND="yask"
                         DEVITO_OPENMP="0"
                         YC_CXX="g++-7"
                     }
                     steps {
                         cleanWorkspace()
                         condaInstallDevito()
                         installYask()
                         runCondaTests()
                         runCodecov()
                         buildDocs()
                     }
                }
                stage('Build and test gcc-7 OPS container') {
                     agent { dockerfile { label 'azure-linux'
                                          filename 'Dockerfile.jenkins'
                                          additionalBuildArgs "--build-arg gccvers=7" } }
                     environment {
                         HOME="${WORKSPACE}"
                         DEVITO_BACKEND="ops"
                     }
                     steps {
                         cleanWorkspace()
                         condaInstallDevito()
                         runCondaTests()
                         runCodecov()
                         buildDocs()
                     }
                }
                stage('Build and test gcc-8 container') {
                     agent { dockerfile { label 'azure-linux-8core'
                                          filename 'Dockerfile.jenkins'
                                          additionalBuildArgs "--build-arg gccvers=8" } }
                     environment {
                         HOME="${WORKSPACE}"
                         DEVITO_OPENMP=0
                     }
                     steps {
                         cleanWorkspace()
                         condaInstallDevito()
                         runCondaTests()
                         runExamples()
                         runCodecov()
                         buildDocs()
                     }
                }
            }
        }
    }
}

def cleanWorkspace() {
    sh "git clean -f -d"
    sh "rm -rf ${WORKSPACE}/scratch"
}

def condaInstallDevito () {
    sh 'conda env create -q -f environment.yml'
    sh 'source activate devito ; pip install -e . ; pip install pytest-xdist ; conda list'
    sh 'source activate devito ; flake8 --exclude .conda,.git --builtins=ArgumentError .'
}

def pipInstallDevito () {
    sh "mkdir -p ${WORKSPACE}/lib/python3.6/site-packages/"
    sh "python setup.py install --prefix=${WORKSPACE}"
    sh 'bin/flake8 --exclude lib/python3.6/site-packages,.git --builtins=ArgumentError .'
}

def installYask () {
    sh "mkdir -p $HOME/.ssh/"
    sh """echo -e "Host github.com\n\tStrictHostKeyChecking no\n" >> $HOME/.ssh/config"""
    sh "source activate devito ; conda install swig"
    sh "mkdir ${WORKSPACE}/scratch"
    dir ("${WORKSPACE}/scratch") { sh 'git clone https://github.com/opesci/yask.git' }
    dir ("${WORKSPACE}/scratch/yask") {
        sh '''source activate devito
              make compiler-api
              pip install -e .
           '''
    }
}

def runPipTests() {
    sh 'python setup.py test'
}

def runCondaTests() {
    sh 'source activate devito ; py.test --cov devito tests/'
}

def runExamples () {
    sh 'source activate devito ; python examples/seismic/benchmark.py test -P tti -so 4 -a -d 20 20 20 -n 5'
    sh 'source activate devito ; python examples/seismic/benchmark.py test -P acoustic -a'
    sh 'source activate devito ; python examples/seismic/acoustic/acoustic_example.py --full'
    sh 'source activate devito ; python examples/seismic/acoustic/acoustic_example.py --full --checkpointing'
    sh 'source activate devito ; python examples/seismic/acoustic/acoustic_example.py --constant --full'
    sh 'source activate devito ; python examples/misc/linalg.py mat-vec mat-mat-sum transpose-mat-vec'
    sh 'source activate devito ; python examples/seismic/tti/tti_example.py -a'
    sh 'source activate devito ; python examples/seismic/tti/tti_example.py -a --noazimuth'
    sh 'source activate devito ; python examples/seismic/elastic/elastic_example.py'
    sh 'source activate devito ; py.test --nbval examples/cfd'
    sh 'source activate devito ; py.test --nbval examples/seismic/tutorials/0[1-3]*'
    sh 'source activate devito ; py.test --nbval examples/compiler'
}

def runCodecov() {
    sh 'source activate devito; codecov'
}

def buildDocs() {
    sh '''source activate devito
          sphinx-apidoc -f -o docs/ examples
          sphinx-apidoc -f -o docs/ devito devito/yask/*
          cd docs
          make html
       '''
}
