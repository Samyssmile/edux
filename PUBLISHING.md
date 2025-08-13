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

### Required Secrets

The following GitHub secrets must be configured:

1. **MAVEN_CENTRAL_USERNAME**: Username from Central Portal token
2. **MAVEN_CENTRAL_PASSWORD**: Password from Central Portal token  
3. **SIGNING_KEY**: GPG private key for artifact signing
4. **SIGNING_PASSWORD**: GPG key passphrase

### Central Portal Token Generation

1. Go to https://central.sonatype.com/account
2. Generate a user token
3. Use the generated username/password for `MAVEN_CENTRAL_USERNAME` and `MAVEN_CENTRAL_PASSWORD`

### Publishing Process

#### Automated (GitHub Actions)
- Automatic publishing occurs on pushes to `main` branch
- GitHub Action runs: `./gradlew publishAllPublicationsToNmcpMavenJavaRepository`
- With `USER_MANAGED` setting, releases require manual approval on Central Portal

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

### Troubleshooting

- Ensure all required secrets are configured in GitHub repository settings
- Verify GPG signing key is properly formatted (include `-----BEGIN PGP PRIVATE KEY BLOCK-----` headers)
- Check Central Portal deployment status at https://central.sonatype.com/