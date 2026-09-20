pipeline {
	agent any
	environment {
		IMAGE_TAG = 'latest'
	}
	options {
        withFolderProperties()
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

		stage('Capture commit message') {
		    steps {
		        script {
		            env.GIT_COMMIT_MESSAGE = sh(
		                script: 'git log -1 --pretty=%B',
		                returnStdout: true
		            ).trim()
		        }
		    }
		}

		stage('Build API') {
			steps {
				sh "echo 'BUILD_TIMESTAMP = \"$BUILD_TIMESTAMP\"' > ./build_info.py"
				sh "echo BUILD_NUMBER = $BUILD_NUMBER >> ./build_info.py"
				sh "git submodule update --init --recursive model_service_common"
				sh "docker buildx use mybuilder"
				sh "docker buildx build --platform linux/amd64 --tag ${DOCKER_REGISTRY}/epa/predictor_models:${IMAGE_TAG} --push ."
			}
		}

		stage('Deploy') {
			steps {
				withKubeConfig([credentialsId: 'k8s', serverUrl: "$K8S_API"]) {
					sh "kubectl rollout restart deployment predictor-models -n $K8S_NAMESPACE"
					sh "kubectl rollout status deployment predictor-models -n $K8S_NAMESPACE --timeout=10m"
				}
			}
		}
	}

	post {
		success {
			emailext (
				to: "tkachenko.valery@gmail.com",
				subject: "${currentBuild.result}: ${JOB_NAME} - Build #${BUILD_NUMBER}",
				body: "Build ${BUILD_NUMBER} of ${env.JOB_NAME} ${currentBuild.result}: ${JOB_NAME}: ${BUILD_URL}\n\nCommit message:\n${env.GIT_COMMIT_MESSAGE ?: 'Unavailable'}",
				attachLog: true,
			)
		}
		failure {
			emailext (
				to: "tkachenko.valery@gmail.com",
				recipientProviders: [
					[$class: 'RequesterRecipientProvider'],
					[$class: 'CulpritsRecipientProvider']
				],
				subject: "${currentBuild.result}: ${JOB_NAME} - Build #${BUILD_NUMBER}",
				body: "Build ${env.BUILD_NUMBER} of ${env.JOB_NAME} failed: ${BUILD_URL}. Please investigate.\n\nCommit message:\n${env.GIT_COMMIT_MESSAGE ?: 'Unavailable'}",
				attachLog: true,
			)
		}
	}
}
