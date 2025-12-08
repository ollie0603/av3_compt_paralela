plugins {
    java
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jocl:jocl:2.0.5")
}

application {
    mainClass.set("Main") 
}

tasks.withType<JavaExec> {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    
    standardInput = System.`in`
}
