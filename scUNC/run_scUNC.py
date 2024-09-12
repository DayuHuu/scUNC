from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import argparse
import load_data as loader
from datasets import TrainDataset
from model import Network
from utils import *

def scUNC_training(X,Y, n_clusters_current, dip_merge_threshold, cluster_loss_weight,ae_weight_loss, centers_cpu, cluster_labels_cpu,
                       dip_matrix_cpu, n_clusters_max, n_clusters_min, dedc_epochs, optimizer, loss_fn, autoencoder,
                       device, dataloader, debug):
    i = 0
    while i < dedc_epochs:
        centers_torch = []
        cluster_labels_torch = torch.from_numpy(cluster_labels_cpu).long().to(device)
        for v in range(2):
            a = torch.from_numpy(centers_cpu[v]).float().to(device)
            centers_torch.append(a)
        dip_matrix_torch = torch.from_numpy(dip_matrix_cpu).float().to(device)
        dip_matrix_eye = dip_matrix_torch + torch.eye(n_clusters_current, device=device)
        dip_matrix_final = dip_matrix_eye / dip_matrix_eye.sum(1).reshape((-1, 1))

        for batch, ids in dataloader:
            for w in range(2):
                batch[w] = batch[w].to(device)
            embedded = autoencoder.encode(batch)
            out1,out2 = autoencoder.decode(embedded)
            embedded_centers_torch = autoencoder.encode(centers_torch)
            # Reconstruction Loss
            ae_loss = loss_fn(out1, batch[0]) + loss_fn(out2, batch[1])
            # Get distances between points and centers. Get nearest center
            squared_diffs = squared_euclidean_distance(embedded_centers_torch, embedded)
            if i != 0:
                # Update labels
                current_labels = squared_diffs.argmin(1)
            else:
                k = np.array(ids[0])
                current_labels = cluster_labels_torch[k]

            onehot_labels = int_to_one_hot(current_labels, n_clusters_current).float()
            cluster_relationships = torch.matmul(onehot_labels, dip_matrix_final)
            escaped_diffs = cluster_relationships * squared_diffs

            # Normalize loss by cluster distances
            squared_center_diffs = squared_euclidean_distance(embedded_centers_torch, embedded_centers_torch)

            # Ignore zero values (diagonal)
            mask = torch.where(squared_center_diffs != 0)
            masked_center_diffs = squared_center_diffs[mask[0], mask[1]]
            sqrt_masked_center_diffs = masked_center_diffs.sqrt()
            masked_center_diffs_std = sqrt_masked_center_diffs.std() if len(sqrt_masked_center_diffs) > 2 else 0

            # Loss function
            cluster_loss = escaped_diffs.sum(1).mean() * (
                    1 + masked_center_diffs_std) / sqrt_masked_center_diffs.mean()
            cluster_loss *= cluster_loss_weight


            loss = ae_loss * ae_weight_loss + cluster_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Update centers
        embedded_data = encode_batchwise(dataloader, autoencoder, device)


        embedded_centers_cpu = autoencoder.encode(centers_torch).detach().cpu().numpy()
        cluster_labels_cpu = np.argmin(cdist(embedded_centers_cpu, embedded_data), axis=0)
        optimal_centers = np.array([np.mean(embedded_data[cluster_labels_cpu == cluster_id], axis=0) for cluster_id in
                                    range(n_clusters_current)])
        centers_cpu, embedded_centers_cpu = get_nearest_points_to_optimal_centers(X, optimal_centers, embedded_data)

        # Update Dips
        dip_matrix_cpu = get_dip_matrix(embedded_data, embedded_centers_cpu, cluster_labels_cpu, n_clusters_current)

        if debug:
            print(
                "Iteration {0}  (n_clusters = {4}) - reconstruction loss: {1} / cluster loss: {2} / total loss: {3}".format(
                    i, ae_loss.item(), cluster_loss.item(), loss.item(), n_clusters_current))
            print("max dip", np.max(dip_matrix_cpu), " at ",
                  np.unravel_index(np.argmax(dip_matrix_cpu, axis=None), dip_matrix_cpu.shape))


        i += 1

        # Start merging procedure
        dip_argmax = np.unravel_index(np.argmax(dip_matrix_cpu, axis=None), dip_matrix_cpu.shape)

        # Is merge possible?
        if i != 0:
            while dip_matrix_cpu[dip_argmax] >= dip_merge_threshold and n_clusters_current > n_clusters_min:
                if debug:
                    print("Start merging in iteration {0}.\nMerging clusters {1} with dip value {2}.".format(i,
                                                                                                             dip_argmax,
                                                                                                             dip_matrix_cpu[
                                                                                                                 dip_argmax]))
                # Reset iteration and reduce number of cluster
                i = 0
                n_clusters_current -= 1
                cluster_labels_cpu, centers_cpu, embedded_centers_cpu, dip_matrix_cpu = \
                    merge_by_dip_value(X, embedded_data, cluster_labels_cpu, dip_argmax, n_clusters_current,
                                        centers_cpu,  embedded_centers_cpu)
                dip_argmax = np.unravel_index(np.argmax(dip_matrix_cpu, axis=None), dip_matrix_cpu.shape)


        if n_clusters_current == 1:
            if debug:
                print("Only one cluster left")
            break

    return cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder


def get_trained_autoencoder(trainloader, learning_rate, n_epochs, device, optimizer_class, loss_fn,
                            input_dim1,input_dim2, embedding_size, autoencoder_class=Network):

    if judge_system():
        act_fn = torch.nn.ReLU
    else:
        act_fn = torch.nn.LeakyReLU


    autoencoder = autoencoder_class(input_A=input_dim1, input_B=input_dim2, embedding_size=embedding_size,
                                    act_fn=act_fn).to(device)

    optimizer = optimizer_class(autoencoder.parameters(), lr=learning_rate)
    autoencoder.start_training(trainloader, n_epochs, device, optimizer, loss_fn)

    return autoencoder

def _scUNC(X, Y, dip_merge_threshold, cluster_loss_weight, ae_weight_loss, n_clusters_max,
             n_clusters_min, batch_size, learning_rate, pretrain_epochs, dedc_epochs, embedding_size,
               debug, optimizer_class=torch.optim.Adam, loss_fn=torch.nn.MSELoss()):

    device = detect_device()

    dataset = TrainDataset(X, Y)
    dataloader = create_data_loader(dataset,batch_size,init=True, labels=None)




    autoencoder = get_trained_autoencoder(dataloader, learning_rate, pretrain_epochs, device,
                                              optimizer_class, loss_fn, args.view_dims[0],args.view_dims[1], embedding_size,
                                              Network)


    embedded_data = encode_batchwise(dataloader, autoencoder, device)

    # Execute Louvain algorithm to get initial micro-clusters in embedded space
    init_centers, cluster_labels_cpu = get_center_labels(embedded_data, resolution=3.0)

    n_clusters_start=len(np.unique(cluster_labels_cpu))
    print("\n "  "Initialize " + str(n_clusters_start) + "  mirco_clusters \n")

    # Get nearest points to optimal centers
    centers_cpu, embedded_centers_cpu = get_nearest_points_to_optimal_centers(X, init_centers, embedded_data)
    # Initial dip values
    dip_matrix_cpu = get_dip_matrix(embedded_data, embedded_centers_cpu, cluster_labels_cpu, n_clusters_start)

    # Reduce learning_rate from pretraining by a magnitude of 10
    dedc_learning_rate = learning_rate * 0.1
    optimizer = optimizer_class(autoencoder.parameters(), lr=dedc_learning_rate)

    # Start clustering training
    cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder = scUNC_training(X,Y, n_clusters_start,
                                                                                          dip_merge_threshold,
                                                                                          cluster_loss_weight,
                                                                                          ae_weight_loss,
                                                                                          centers_cpu,
                                                                                          cluster_labels_cpu,
                                                                                          dip_matrix_cpu,
                                                                                          n_clusters_max,
                                                                                          n_clusters_min,
                                                                                          dedc_epochs,
                                                                                          optimizer,
                                                                                          loss_fn,
                                                                                          autoencoder,
                                                                                          device,
                                                                                          dataloader,
                                                                                          debug)

    return cluster_labels_cpu, n_clusters_current, centers_cpu, autoencoder


class scUNC():

    def __init__(self, dip_merge_threshold, cluster_loss_weight, ae_loss_weight,  batch_size,
                 learning_rate, pretrain_epochs, dedc_epochs, embedding_size,
                 n_clusters_max, n_clusters_min, debug):

        self.dip_merge_threshold = dip_merge_threshold
        self.cluster_loss_weight = cluster_loss_weight
        self.ae_loss_weight=ae_loss_weight
        self.n_clusters_max = n_clusters_max
        self.n_clusters_min = n_clusters_min
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pretrain_epochs = pretrain_epochs
        self.dedc_epochs = dedc_epochs
        self.embedding_size = embedding_size
        self.debug = debug

    def fit(self, X,Y):
        labels, n_clusters, centers, autoencoder = _scUNC(X,Y, self.dip_merge_threshold,
                                                               self.cluster_loss_weight,
                                                               self.ae_loss_weight,
                                                               self.n_clusters_max,
                                                               self.n_clusters_min,
                                                               self.batch_size,
                                                               self.learning_rate,
                                                               self.pretrain_epochs,
                                                               self.dedc_epochs,
                                                               self.embedding_size,
                                                               self.debug)

        self.labels_ = labels
        self.n_clusters_ = n_clusters
        self.cluster_centers_ = centers
        self.autoencoder = autoencoder

        return labels, n_clusters


if __name__ == "__main__":
    my_data_dic = loader.ALL_data
    for i_d in my_data_dic:
        data_para = my_data_dic[i_d]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        parser = argparse.ArgumentParser(description='scUNC')
        parser.add_argument('--dataset', default=data_para)
        parser.add_argument("--view_dims", default=data_para['n_input'])
        parser.add_argument('--name', type=str, default=data_para[1])
        parser.add_argument('--dip_merge_threshold', type=float, default=1)
        parser.add_argument('--cluster_loss_weight', type=float, default=0.1)
        parser.add_argument('--ae_loss_weight', type=float, default=100)
        parser.add_argument('--n_clusters_max', type=int, default=np.inf)
        parser.add_argument('--n_clusters_min', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--learning_rate ', type=float, default=1e-4)
        parser.add_argument('--pretrain_epochs', type=float, default=100)
        parser.add_argument('--dedc_epochs', type=float, default=50)
        parser.add_argument('--embedding_size ', type=float, default=100)
        parser.add_argument('--debug', type=bool, default=True)
        args = parser.parse_args()
        X, Y = loader.load_data(args.dataset)
        labels = Y[0].copy().astype(np.int32)


        # Training
        myscUNC = scUNC(dip_merge_threshold=args.dip_merge_threshold, cluster_loss_weight=args. cluster_loss_weight,ae_loss_weight=args.ae_loss_weight,batch_size=args.batch_size,
                        learning_rate=1e-4,pretrain_epochs=args.pretrain_epochs,dedc_epochs=args.dedc_epochs,
                        embedding_size=100, n_clusters_max=args.n_clusters_max,
                        n_clusters_min=args.n_clusters_min, debug=args.debug)

        cluster_labels, estimated_cluster_numbers = myscUNC.fit(X,Y)

        # === Print results ===
        ari = float(np.round(ari_score(labels, cluster_labels), 4))
        nmi = float(np.round(nmi_score(labels, cluster_labels), 4))
        acc = float(np.round(cluster_acc(labels, cluster_labels), 4))
        pur = float(np.round(Purity_score(labels, cluster_labels), 4))
        
        
        print("The estimated number of clusters:", estimated_cluster_numbers)
        print("ARI: ", ari)
        print("NMI:", nmi)
        print("ACC: ", acc)
        print("PUR: ", pur)
        




