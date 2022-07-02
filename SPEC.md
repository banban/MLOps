# Containerization of ML model
A bunch of myocardial infarction risk predicting ML models released by Docker container as a web service. The requirements started with a predictive model of myocardial infarction to additional requirements such as 30-day death prediction and CABG/Intervention prediction at this point. It is believed that the requirements will keep growing. This project aims to spend minimal time in adjusting or creating a new docker container every time a new ML model is built and be able to bulk process capacity if a user chooses to send package containing an arbitrary number of patient records.

## Diagrams
We actively use [Mermaid](https://mermaidjs.github.io) and [Yed](https://www.yworks.com/products/yed), [PlantUml](https://plantuml.com/) diagrams to enforce our documentation in markup format. All produced diagrams can be found in **/images/** folder. 
To read/update/render our Mermaid diagrams please use [live editor](https://mermaid.live) or VS Code Markdown extensions: [Mermaid how to](https://www.youtube.com/watch?v=JiQmpA474BY),[PlantUml how to](https://www.youtube.com/watch?v=tPh9_Cx4yZY). Also, Mermaid diagrams are automatically rendered in browser for [Microsoft DevOps](https://dev.azure.com/) wiki pages.


**Flow Diagrams**
``` mermaid
 graph LR;
 A1[Demonstrate]--> B1[Validate]-->C1[Remediate]-->A1[Demonstrate];
 A2[Document]-->B2[Build]-->C2[Test]-->B2[Build];
```

**Docker Diagram**
``` mermaid
 graph LR;
 Dk1[Dockerfile]
 Dk2[Docker-compose.yml]
 Dk3[Image]
 Dk4[Container]
 Dk5[Volume]
 
 Dk1-->|build|Dk3
 Dk2-->|compose up|Dk3
 Dk3-->|deploy|Dk4
 Dk5-->|mount|Dk4

```

**API Diagram**
``` mermaid
 graph TD;
 A3[Request]-->B3{API}-->|REST|C3[Response];
 B3-->|GraphQL|C3
 B3-->|gRPC|C3
```

**Prediction Diagram**
``` mermaid
graph TD;
 A4[input]-->B4{predict};
 C41[outcome]
 C42[event]
 C43[revasc]
 B4-->|dl|C41
 B4-->|xgb|C41
 B4-->|xgb|C42
 B4-->|dl|C43
 B4-->|xgb|C43
```

**Use Case Diagram**
``` mermaid
erDiagram
    MLDesigner ||--o{Model: uploads
    SystemManager ||--o{Model: deploys
    Model ||--|{Parameter: contains
    Model {
        Type type
        String version
        String name
        String description
        binary content
    }
    TabularData{
        svc TroponinResults
    }
    User ||--o{Model: uses
    Model ||--o{TabularData: uses
```

**Sequence Diagram**
``` mermaid
sequenceDiagram
    autonumber
    participant AIML
    participant Team10
    participant HeartAI
    participant RAPIDx
    Team10->>Team10: Build API
    AIML->>Team10: Upload models and API
    Note right of AIML: Serialized model .pickle<br/>structured data(tropoline results)?
    activate Team10 
    Team10->>Team10: Build Docker images
    Team10->>Team10: Test Docker containers
    Team10-->>AIML: Test results
    deactivate Team10
    AIML->>HeartAI: Upload Docker images
    HeartAI->>HeartAI: Deploy Docker images<br/> as containers
    loop Model inference
        RAPIDx->>HeartAI: Input data<br/>patient record
        activate HeartAI
        HeartAI-->>RAPIDx: Predict results<br/>probability values
        deactivate HeartAI
    end
```

**Entity Relationship Diagram**
``` mermaid
erDiagram
    PatientRecord{
        Int age
        Enum gender
        Array angiogram 
        Array troponineHistory
    }
    Category {
        String code
        String name
    }
    Model {
        Enum category
        String version
        String name
        String description
        Binary content
    }
    Troponin {
        DateTime timestamp
        Int value
    }
    InferenceResult{
        Int inferenceId
        Array featureArray
    }
    InferenceFeature{
        String name
        Int value
    }
    PatientRecord ||..o{Troponin: contains
    Category ||--o{Model: references
    Inference ||--|{PatientRecord: accepts
    Inference ||--o{Model: invokes
    Model ||..|{Parameter: contains
    Model ||--o{PatientRecord: uses
    Model ||--o{InferenceResult: predicts
    InferenceResult ||--o{InferenceFeature: contains
```

**Component Diagram**
```plantuml
@startuml
!unquoted procedure COMP_TEXTGENCOMP(ifc, name)
[name] << Comp >>
interface Ifc << ifc >> AS name##Ifc
name##Ifc - [name]
!endprocedure
COMP_TEXTGENCOMP(DL, v3)
COMP_TEXTGENCOMP(XGB, v4)
COMP_TEXTGENCOMP(DL+XGB, API_SERVER)
@enduml
```