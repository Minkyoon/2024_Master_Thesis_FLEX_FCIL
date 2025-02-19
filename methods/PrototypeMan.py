import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import pickle



class PrototypeManager:
    def __init__(self, args):
        self.args = args
        self.prototypes = {}
        self.class_counts = {}
        self.pseudo_feature_path=os.path.join(args['pseudo_feature_path'],args['exp_name'])

    def generate_prototypes(self, model, loader):
        """Generate prototypes for each class using the given model and data loader."""
        model.eval()
        all_features, all_labels = [], []

        with torch.no_grad():
            for _, (index, images, labels) in enumerate(loader):
                images, labels = images.cuda(), labels.cuda()
                features = model(images)['features']
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())

        all_features = torch.cat(all_features, 0)
        all_labels = torch.cat(all_labels, 0)

        # Compute prototypes for each class
        for label in torch.unique(all_labels):
            class_features = all_features[all_labels == label]
            self.prototypes[label.item()] = class_features.mean(0)
            self.class_counts[label.item()] = class_features.size(0)
            
        
            
    def generate_prototypes2(self, model, images, labels):
        """Generate prototypes for each class using the given model and data loader."""
        model.eval()
        all_features, all_labels = [], []


        images, labels = images.cuda(), labels.cuda()
        features = model(images)['features']
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())

        all_features = torch.cat(all_features, 0)
        all_labels = torch.cat(all_labels, 0)

        # Compute prototypes for each class
        for label in torch.unique(all_labels):
            class_features = all_features[all_labels == label]
            self.prototypes[label.item()] = class_features.mean(0)
            self.class_counts[label.item()] = class_features.size(0)
            
        return all_features, all_labels 
    

    def aggregate_prototypes(self):
        """Aggregate prototypes from all clients by averaging them."""
        global_prototypes = {}
        total_class_counts = {}

        for class_id, proto in self.prototypes.items():  # Assuming self.prototypes is a dict of tensors
            if class_id not in global_prototypes:
                global_prototypes[class_id] = proto * self.class_counts[class_id]
                total_class_counts[class_id] = self.class_counts[class_id]
            else:
                global_prototypes[class_id] += proto * self.class_counts[class_id]
                total_class_counts[class_id] += self.class_counts[class_id]

        # Finalize the global prototypes by averaging
        for class_id in global_prototypes:
            global_prototypes[class_id] /= total_class_counts[class_id]

        return global_prototypes, total_class_counts

    def send_to_clients(self, prototypes, class_counts):
        """Send aggregated prototypes back to clients."""
        for class_id, proto in prototypes.items():
            if class_id in self.prototypes:
                # 기존 프로토타입에 새로 받은 프로토타입을 더함
                self.prototypes[class_id] += proto
                self.class_counts[class_id] += class_counts[class_id]
            else:
                # 새로운 클래스에 대해 프로토타입과 개수를 초기화
                self.prototypes[class_id] = proto
                self.class_counts[class_id] = class_counts[class_id]

    def get_prototypes(self):
        """Return the current global prototypes."""
        return self.prototypes

    def compute_prototype_loss(self, features, labels, prototypes):
        """Compute the loss between the features and the prototypes for each class."""
        loss = 0
        for i, label in enumerate(labels):
            proto = prototypes[label.item()]
            loss += F.mse_loss(features[i], proto)
        return loss / len(labels)


    def mahalanobis_distance(self, prototype, other_prototypes, covariance_matrix):
        """
        Compute the Mahalanobis distance between a prototype and a set of other prototypes.
        :param prototype: a prototype tensor of shape (D,)
        :param other_prototypes: a tensor of prototypes of shape (N, D)
        :param covariance_matrix: a covariance matrix of shape (D, D)
        :return: a tensor of distances of shape (N,)
        """
        delta = other_prototypes - prototype
        inv_covmat = torch.inverse(covariance_matrix)
        dist = torch.sqrt(torch.diag(delta @ inv_covmat @ delta.T))
        return dist    

    def augment_prototypes(self, prototypes, labels, alpha_range=(0.5, 0.8)):
        """
        Augment features using prototypes and labels.
        :param prototypes: dictionary of class prototypes
        :param labels: tensor of labels for the current batch
        :param alpha_range: range of alpha values for interpolation
        :return: augmented features and labels
        """
        augmented_features = []
        augmented_labels = []

        # Iterate over the batch of labels
        for label in labels:
            class_proto = prototypes[label.item()]  # Prototype of the class

            # Randomly choose 3 other prototypes for augmentation
            other_classes = torch.multinomial(torch.ones(len(prototypes)), 3, replacement=False)
            other_protos = [prototypes[cls.item()] for cls in other_classes if cls.item() != label.item()]

            # Interpolate between the original prototype and the chosen prototypes
            for other_proto in other_protos:
                # Randomly select alpha from the range
                alpha = torch.rand(1).item() * (alpha_range[1] - alpha_range[0]) + alpha_range[0]
                augmented_feature = alpha * class_proto + (1 - alpha) * other_proto
                augmented_features.append(augmented_feature)
                augmented_labels.append(label)

        augmented_features = torch.stack(augmented_features)
        augmented_labels = torch.tensor(augmented_labels)

        return augmented_features, augmented_labels
    
    
    def extract_features(self, model, loader):
        """Extract features and labels from the model using the given data loader."""
        model.eval()
        all_features, all_labels = [], []

        with torch.no_grad():
            for _, (_, images, labels) in enumerate(loader):
                images, labels = images.cuda(), labels.cuda()
                features = model(images)['features']
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())

        all_features = torch.cat(all_features, 0)
        all_labels = torch.cat(all_labels, 0)

        return all_features, all_labels

    def generate_pseudo_features(self, current_features, current_labels, task_index, idx):
        """Generate pseudo-features using the prototypes for each class."""
        current_labels = current_labels.cpu()
        current_features = current_features.cpu()
        new_class_means = {label: self.prototypes[label] for label in np.unique(current_labels)}
        pseudo_features = []
        pseudo_labels = []
        old_class_labels = [label for label in self.prototypes.keys() if label not in new_class_means.keys()]
        num_data_points = len(np.unique(current_labels)) + 1
        n_components = min(num_data_points, 512)

        # 각 과거 클래스에 대해 수도 특성 생성
        for old_label in old_class_labels:
            if old_label in self.prototypes:
                old_class_mean = self.prototypes[old_label]

                # 가장 유사한 새로운 클래스 찾기
                closest_new_label = self.find_closest_class(new_class_means, old_class_mean)
                new_class_mean = new_class_means[closest_new_label]

                old_class_mean_reshaped = old_class_mean.reshape(1, -1)
                new_class_means_array = np.array([mean.detach().cpu().numpy().reshape(1, -1) for mean in new_class_means.values()])

                # PCA 학습
                pca_input = np.vstack([old_class_mean_reshaped] + list(new_class_means_array))
                pca = PCA(n_components=n_components)
                pca.fit(pca_input)

                principal_axes = pca.components_
                difference_vector = old_class_mean_reshaped.detach() - new_class_mean.detach().cpu().numpy().reshape(1, -1)
                projected_difference = np.dot(np.dot(difference_vector, principal_axes.T), principal_axes)
                projected_difference = projected_difference.reshape(-1)

                # 선택된 새로운 클래스의 모든 특성에 대해 수도 특성 생성
                for feature in current_features[current_labels == closest_new_label]:
                    pseudo_feature = feature + projected_difference
                    pseudo_features.append(pseudo_feature)
                    pseudo_labels.append(old_label)

        self.save_pseudo_features(current_features,current_labels,pseudo_features, pseudo_labels, idx,task_index,)
        
        
    def generate_pseudo_features2(self, current_features, current_labels, task_index, idx):
        """Generate pseudo-features using the prototypes for each class."""
        current_labels = current_labels.cpu()
        current_features = current_features.cpu()
        current_features=current_features.detach().numpy()
        new_class_means = {label: self.prototypes[label] for label in np.unique(current_labels)}
        pseudo_features = []
        pseudo_labels = []
        old_class_labels = list(range(10 * task_index))
        num_data_points = len(np.unique(current_labels)) + 1
        n_components = min(num_data_points, 512)

        # 각 과거 클래스에 대해 수도 특성 생성
        for old_label in old_class_labels:
            if old_label in self.prototypes:
              
                old_class_mean = self.prototypes[old_label]

                # 가장 유사한 새로운 클래스 찾기

                closest_new_label = self.find_closest_class(new_class_means, old_class_mean)
                new_class_mean = new_class_means[closest_new_label]

                old_class_mean_reshaped = old_class_mean.detach().cpu().numpy().reshape(1, -1)
                new_class_means_array = np.array([mean.detach().cpu().numpy().reshape(1, -1) for mean in new_class_means.values()])

                # PCA 학습
                pca_input = np.vstack([old_class_mean_reshaped] + list(new_class_means_array))
                pca = PCA(n_components=n_components)
                pca.fit(pca_input)

                principal_axes = pca.components_
                difference_vector = old_class_mean_reshaped - new_class_mean.detach().cpu().numpy().reshape(1, -1)
                projected_difference = np.dot(np.dot(difference_vector, principal_axes.T), principal_axes)
                projected_difference = projected_difference.reshape(-1)
                

                # 선택된 새로운 클래스의 모든 특성에 대해 수도 특성 생성
                for feature in current_features[current_labels == closest_new_label]:
                    pseudo_feature = feature + projected_difference
                    pseudo_features.append(pseudo_feature)
                    pseudo_labels.append(old_label)
             
  

        return current_features, current_labels ,pseudo_features, pseudo_labels

    def find_closest_class(self, new_class_means, old_class_mean):
        """Find the closest new class to the old class using cosine similarity."""
        closest_label = None
        highest_similarity = -np.inf
        old_class_mean = old_class_mean.flatten().detach().cpu().numpy()
        
        
        for label, mean in new_class_means.items():
            mean = mean.flatten().detach().cpu().numpy()
            similarity = np.dot(mean, old_class_mean) / (np.linalg.norm(mean) * np.linalg.norm(old_class_mean))
            if similarity > highest_similarity:
                highest_similarity = similarity
                closest_label = label
        return closest_label

    def save_pseudo_features(self, current_features, current_labels, pseudo_features, pseudo_labels, idx,task_index):
        """Save current features, current labels, and generated pseudo-features to a file."""
        save_path = os.path.join(self.pseudo_feature_path,f'client_{idx}',f'pseudo_features_task_{task_index}.pkl')
        os.makedirs(os.path.join(self.pseudo_feature_path,f'client_{idx}'), exist_ok=True)
        
        # 데이터 저장
        with open(save_path, 'wb') as f:
            pickle.dump({
                'current_features': current_features,
                'current_labels': current_labels,
                'pseudo_features': pseudo_features,
                'pseudo_labels': pseudo_labels
            }, f)

    def load_pseudo_features(self, task_index,client_idx):
        """Load saved current features, current labels, and pseudo-features from a file."""
        load_path = os.path.join(self.pseudo_feature_path, f'client_{client_idx}' ,f'pseudo_features_task_{task_index}.pkl')
        
        # 데이터 불러오기
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            
        current_features = data['current_features']
        current_labels = data['current_labels']
        pseudo_features = data['pseudo_features']
        pseudo_labels = data['pseudo_labels']
        
        return current_features, current_labels, pseudo_features, pseudo_labels