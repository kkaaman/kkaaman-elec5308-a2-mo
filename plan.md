# Title
## Project Goal

## Dataset
The team expects to use a combination of publicly available autonomous-driving datasets and data generated from the selected simulator. The dataset will primarily consist of RGB road-scene images with task-specific ground-truth annotations, such as object bounding boxes, semantic segmentation masks, and lane or drivable-area labels. This supports the required perception capabilities of lane and driving-area detection, object detection and tracking, semantic segmentation, and traffic-condition detection.

Public datasets such as Cityscapes may be used where appropriate to provide labelled real-world road scenes, particularly for segmentation tasks. Simulator-generated data can then be used to supplement these datasets with scenes that more closely reflect the environment in which the final system will be demonstrated. The final dataset selection will depend on the perception methods chosen by the team, the quality and compatibility of available annotations, and the computational cost of training and evaluation.

For any learned components, the data will be divided into training, validation and test sets. The training set will be used to optimise model parameters, the validation set will support hyperparameter tuning and model selection, and the test set will remain isolated until final evaluation. Care will be taken to avoid data leakage between these splits, particularly where sequential simulator frames may contain highly similar scenes.

The following assumptions are currently being made:

- RGB camera input will provide sufficient information for the baseline perception requirements.
- Suitable annotations will be available either from public datasets or directly from the selected simulator.
- The dataset will provide adequate representation of important classes, including roads, lane markings, vehicles, pedestrians and traffic signs.
- Class imbalance is likely to occur, particularly for small or infrequent classes, and will need to be considered during training and evaluation.
- Training and evaluation data should cover a sufficiently broad range of road layouts, lighting conditions, weather conditions, object scales and levels of occlusion.
- Simulator-generated data may not perfectly reflect real-world visual distributions, so conclusions drawn from simulator performance should be interpreted within that limitation.
- Dataset loading and preprocessing will be kept separate from simulator-specific functionality where practical to support the simulator-agnostic architecture required by the project.

# I like apples and bananas

## Risks
The project introduces several technical risks that may affect the reliability, generalisation and real-time performance of the final autonomous perception and navigation system.

- **Distribution mismatch:** Performance may degrade when the operating environment differs from the data used during development. Changes in weather, lighting, road layout, visual texture or simulator environment may produce a different input distribution from the training data. This risk will be reduced by using varied datasets and evaluating the system across multiple operating conditions rather than relying on performance from a single environment.

- **Class imbalance and misleading evaluation:** Some important classes, such as lane markings, pedestrians and traffic signs, may represent only a small proportion of the available data. High aggregate accuracy may therefore hide poor performance on safety-relevant classes. Evaluation will use task-appropriate metrics such as precision, recall, IoU and class-level performance where relevant, rather than relying solely on overall accuracy.

- **Overfitting and data leakage:** Models may perform strongly on familiar data while failing to generalise to unseen scenes. This risk is increased if highly similar frames from the same driving sequence are distributed across training and test sets. Training, validation and test data will therefore be kept appropriately separated, with the test set reserved for final evaluation.

- **Difficult perception conditions:** Perception performance may deteriorate under conditions such as rain, low illumination, shadows, reflections, occlusion, faded lane markings and visually complex backgrounds. These cases will be included in evaluation where practical so that limitations are identified rather than hidden by average-case performance.

- **Small and distant object detection:** Pedestrians, vehicles and traffic signs may become difficult to detect when they occupy only a small number of pixels. This may reduce recall for objects that are still operationally important. Detection results should therefore be considered across different object scales where possible.

- **Real-time performance and latency:** More computationally complex models may improve perception accuracy but increase inference latency. Since perception outputs are used by downstream navigation and control components, excessive latency may reduce overall system responsiveness. Model selection will therefore need to balance predictive performance with computational cost and achievable frame rate.

- **Simulator dependence:** Components that depend directly on simulator-specific APIs may be difficult to reuse or extend. This would conflict with the required simulator-agnostic core architecture. Simulator interaction will therefore be isolated behind adapters, with core perception and navigation components operating on general data structures.

- **Integration risk:** Modules that perform correctly in isolation may still fail when connected due to inconsistent assumptions, data formats, coordinate systems or timing behaviour. This will be mitigated through clearly defined interfaces, modular testing and incremental integration rather than delaying integration until the end of development.

- **Reproducibility risk:** Differences in package versions, model configuration, preprocessing or dataset preparation may produce inconsistent results across team members or machines. Dependencies, configuration and relevant experimental assumptions will therefore be documented and version-controlled where practical.

## Success Criteria

## Project Goal (Mohammad)
