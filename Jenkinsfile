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
		            env.GIT_COMMIT_SUMMARY = env.GIT_COMMIT_MESSAGE.replaceAll(/\s+/, ' ').take(50)
		            env.GIT_COMMIT_AUTHOR = sh(script: "git log -1 --format='%an <%ae>'", returnStdout: true).trim()
		        }
		    }
		}

		stage('Build API') {
			steps {
				sh "echo 'BUILD_TIMESTAMP = \"$BUILD_TIMESTAMP\"' > ./build_info.py"
				sh "echo BUILD_NUMBER = $BUILD_NUMBER >> ./build_info.py"
				sh "git submodule sync --recursive"
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
			script {
			    def commitDiff = 'Unavailable'
			    if (env.GIT_COMMIT_MESSAGE) {
			        try {
			            commitDiff = sh(script: 'git show --first-parent --format= --no-ext-diff --no-color HEAD', returnStdout: true).trim()
			        } catch (Exception ignored) {
			            echo 'Could not read commit diff for email.'
			        }
			    }
			    def commitAuthorHtml = (env.GIT_COMMIT_AUTHOR ?: 'Unavailable').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
			    def commitMessageHtml = (env.GIT_COMMIT_MESSAGE ?: 'Unavailable').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
			    def commitDiffHtml = (commitDiff ?: 'No changes').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
			    emailext (
				    to: "tkachenko.valery@gmail.com",
				    subject: "${currentBuild.result}: ${JOB_NAME} - ${env.GIT_COMMIT_SUMMARY ?: 'Unavailable'}",
				    mimeType: 'text/html',
				    body: """
				        <h2>${env.JOB_NAME}: ${currentBuild.currentResult}</h2>
				        <p>Build ${env.BUILD_NUMBER}: <a href="${env.BUILD_URL}">Open build</a></p>
				        <p><strong>Commit author:</strong> ${commitAuthorHtml}</p>
				        <h3>Commit message</h3>
				        <pre style="font-family:Consolas,Menlo,monospace;white-space:pre-wrap">${commitMessageHtml}</pre>
				        <h3>Commit diff</h3>
				        <pre style="font-family:Consolas,Menlo,monospace;font-size:12px;line-height:1.4;white-space:pre;overflow-x:auto;background:#f6f8fa;padding:12px;border:1px solid #ddd">${commitDiffHtml}</pre>
				    """.stripIndent().trim(),
				    attachLog: true,
			    )
			}
		}
		failure {
			script {
			    def commitDiff = 'Unavailable'
			    if (env.GIT_COMMIT_MESSAGE) {
			        try {
			            commitDiff = sh(script: 'git show --first-parent --format= --no-ext-diff --no-color HEAD', returnStdout: true).trim()
			        } catch (Exception ignored) {
			            echo 'Could not read commit diff for email.'
			        }
			    }
			    def commitAuthorHtml = (env.GIT_COMMIT_AUTHOR ?: 'Unavailable').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
			    def commitMessageHtml = (env.GIT_COMMIT_MESSAGE ?: 'Unavailable').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
			    def commitDiffHtml = (commitDiff ?: 'No changes').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
			    emailext (
				    to: "tkachenko.valery@gmail.com",
				    recipientProviders: [
					    [$class: 'RequesterRecipientProvider'],
					    [$class: 'CulpritsRecipientProvider']
				    ],
				    subject: "${currentBuild.result}: ${JOB_NAME} - ${env.GIT_COMMIT_SUMMARY ?: 'Unavailable'}",
				    mimeType: 'text/html',
				    body: """
				        <h2>${env.JOB_NAME}: ${currentBuild.currentResult}</h2>
				        <p>Build ${env.BUILD_NUMBER}: <a href="${env.BUILD_URL}">Open build</a></p>
				        <p><strong>Commit author:</strong> ${commitAuthorHtml}</p>
				        <h3>Commit message</h3>
				        <pre style="font-family:Consolas,Menlo,monospace;white-space:pre-wrap">${commitMessageHtml}</pre>
				        <h3>Commit diff</h3>
				        <pre style="font-family:Consolas,Menlo,monospace;font-size:12px;line-height:1.4;white-space:pre;overflow-x:auto;background:#f6f8fa;padding:12px;border:1px solid #ddd">${commitDiffHtml}</pre>
				    """.stripIndent().trim(),
				    attachLog: true,
			    )
			}
		}
	}
}
