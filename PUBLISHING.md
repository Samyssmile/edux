# Publishing to Maven Central

This document describes the new publishing process for EDUX library to Maven Central using the new Central Portal API.

## Background

As of November 2024, OSSRH (OSS Repository Hosting) reached end of life and was shut down. We have migrated from the legacy OSSRH staging API to the new Central Portal publishing method.

**Official Statement:** https://central.sonatype.org/pages/ossrh-eol/

## New Publishing Method

We now use the `com.gradleup.nmcp` Gradle plugin, which provides integration with the new Maven Central Portal API.

### Configuration

The publishing configuration is in `lib/build.gradle`:

```gradle
plugins {
    // ... other plugins
    id 'com.gradleup.nmcp' version '0.0.8'
}

nmcp {
    publish("mavenJava") {
        username = System.getenv("MAVEN_CENTRAL_USERNAME")
        password = System.getenv("MAVEN_CENTRAL_PASSWORD")
        publicationType = "USER_MANAGED"  // Manual release control
    }
}
```

### Required Environment Variables

For local development and CI/CD, the following environment variables must be set:

1. **MAVEN_CENTRAL_USERNAME**: Username from Central Portal token
2. **MAVEN_CENTRAL_PASSWORD**: Password from Central Portal token  
3. **GPG_PRIVATE_KEY**: GPG private key for artifact signing (complete PGP block)
4. **GPG_PASSPHRASE**: GPG key passphrase

### Local Development Setup

Use the provided setup script:
```bash
# Source the environment variables
source setup-publishing-env.sh

# Test local publishing first
./gradlew lib:publishMavenJavaPublicationToMavenLocal

# Publish to Central Portal
./gradlew lib:publish
# or
./gradlew lib:publishAllPublicationsToCentralPortal
```

### Central Portal Token Generation

1. Go to https://central.sonatype.com/account
2. Generate a user token
3. Use the generated username/password for `MAVEN_CENTRAL_USERNAME` and `MAVEN_CENTRAL_PASSWORD`

### Publishing Process

#### Automated (GitHub Actions)
- Automatic publishing occurs on pushes to `main` branch
- GitHub Action runs: `./gradlew publishAllPublicationsToNmcpMavenJavaRepository`
- With `USER_MANAGED` setting, releases require manual approval on Central Portal

**Required GitHub Repository Secrets:**
- `MAVEN_CENTRAL_USERNAME`: Username from Central Portal token
- `MAVEN_CENTRAL_PASSWORD`: Password from Central Portal token  
- `GPG_PRIVATE_KEY`: Complete GPG private key block
- `GPG_PASSPHRASE`: GPG key passphrase

To configure these secrets:
1. Go to your GitHub repository → Settings → Secrets and variables → Actions
2. Click "New repository secret" for each required secret
3. Use the exact names listed above

#### Manual Publishing
```bash
# Publish to Central Portal
./gradlew publishAllPublicationsToNmcpMavenJavaRepository

# Check available tasks
./gradlew tasks --all | grep nmcp
```

### Publication Type Options

- **USER_MANAGED**: Artifacts are uploaded but require manual release approval in Central Portal
- **AUTOMATIC**: Artifacts are automatically published after validation

### Verification

After publishing with `USER_MANAGED`:
1. Log into https://central.sonatype.com/
2. Navigate to deployments
3. Review and approve the deployment for public release

### Migration Changes

- ✅ Removed legacy OSSRH repository configuration
- ✅ Added `com.gradleup.nmcp` plugin
- ✅ Updated GitHub Actions workflow
- ✅ Fixed JUnit test dependencies (unrelated but needed for build)

### GPG Signing Configuration

The build now supports both in-memory GPG signing and GPG command-line tool:

```gradle
signing {
    required { gradle.taskGraph.hasTask("publish") }
    def signingKey = System.getenv("GPG_PRIVATE_KEY")
    def signingPassword = System.getenv("GPG_PASSPHRASE")
    
    if (signingKey && signingPassword) {
        useInMemoryPgpKeys(signingKey, signingPassword)  // Preferred for CI/CD
    } else {
        useGpgCmd()  // Fallback to local GPG installation
    }
    sign publishing.publications.mavenJava
}
```

### Troubleshooting

- **GPG Error "finished with non-zero exit value 2"**: Use the provided `setup-publishing-env.sh` script to set environment variables for in-memory GPG signing
- Ensure all required environment variables are set before publishing
- Verify GPG private key is properly formatted (include `-----BEGIN PGP PRIVATE KEY BLOCK-----` headers)
- Test local publishing first: `./gradlew lib:publishMavenJavaPublicationToMavenLocal`
- Check Central Portal deployment status at https://central.sonatype.com/