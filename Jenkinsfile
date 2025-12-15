pipeline {
	agent any
	environment {
		DOCKER_REGISTRY = 'docker.sciencedataexperts.com'
		IMAGE_TAG = 'latest'
	}

	stages {
		stage('Setup Environment') {
			steps {
				withCredentials([usernamePassword(credentialsId: 'jenkins', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
					sh "docker login -u $USERNAME -p $PASSWORD $DOCKER_REGISTRY"
				}
			}
		}

		stage('SCM') {
			steps {
				git poll: true, branch: 'main', credentialsId: 'valery_tkachenko', url: 'https://bitbucket.org/scidataexperts/pf_python_model_building.git'
			}
		}

		stage('Dependencies check') {
			steps {
				dependencyCheck additionalArguments: "--nvdApiKey ${NVD_API_KEY}", odcInstallation: 'OWASP-Dependency-Check'
				dependencyCheckPublisher pattern: ''
			}
		}

		stage('Build API') {
			steps {
				sh "echo 'BUILD_TIMESTAMP = \"$BUILD_TIMESTAMP\"' > ./build_info.py"
				sh "echo BUILD_NUMBER = $BUILD_NUMBER >> ./build_info.py"

				sh "docker buildx use mybuilder"
				sh "docker buildx build --platform linux/amd64 --tag ${DOCKER_REGISTRY}/epa/predictor_models:${IMAGE_TAG} --push ."
			}
		}

		stage('Deploy') {
			steps {
				withKubeConfig([credentialsId: 'k8s', serverUrl: 'https://k8s.sciencedataexperts.com:6443']) {
					sh "kubectl rollout restart deployment predictor-models -n models-dev"
				}
			}
		}
	}

	post {
		success {
			emailext (
				to: "tkachenko.valery@gmail.com",
				subject: "${currentBuild.result}: ${JOB_NAME} - Build #${BUILD_NUMBER}",
				body: "Build ${BUILD_NUMBER} of ${env.JOB_NAME} ${currentBuild.result}: ${JOB_NAME}: ${BUILD_URL}",
				attachLog: true,
			)
		}
		failure {
			emailext (
				to: "tkachenko.valery@gmail.com, Williams.Antony@epa.gov",
				recipientProviders: [
					[$class: 'RequesterRecipientProvider'],
					[$class: 'CulpritsRecipientProvider']
				],
				subject: "${currentBuild.result}: ${JOB_NAME} - Build #${BUILD_NUMBER}",
				body: "Build ${env.BUILD_NUMBER} of ${env.JOB_NAME} failed: ${BUILD_URL}. Please investigate.",
				attachLog: true,
			)
		}
	}
}
