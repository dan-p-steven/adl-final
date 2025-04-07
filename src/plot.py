import optuna
import optuna.visualization as vis
import plotly.graph_objects as go


def plot_objective_history():
    study = optuna.load_study(study_name='sentiment_lstm_hpo',
                                storage='sqlite:///./models/sentiment_lstm_hpo.db')
        
    # Get all trials and their objective values
    trials = study.trials
    filtered_trials = []

    for t in trials:
        if t.value and t.value < 0.44:
            filtered_trials.append(t)
 
    


    # Create a new filtered study from the filtered trials
    filtered_study = optuna.create_study(study_name='filtered_study', load_if_exists=True)
    filtered_study.add_trials(filtered_trials)

    # Visualize the optimization history without outliers
    fig1 = vis.plot_optimization_history(filtered_study)
    fig1.show()

    fig = optuna.visualization.plot_slice(filtered_study)
    fig.show()

    fig = optuna.visualization.plot_contour(study)
    fig.show()


def contours_plot(study, params, target):
    '''
    Contour plot the target param against all other parameters.
    '''

    params.remove(target)

    for p in params:


        fig = optuna.visualization.plot_contour(study, params=[target, p])
        fig.show()

def plot_param_importances(study):



    fig = vis.plot_param_importances(study)
    fig.show()

    fig = optuna.visualization.plot_slice(study)
    fig.show()

def plot_accuracies_losses(train_stats, val_stats, train_desc, val_desc, title, x_title, y_title):
    epochs = [i for i in range(1, len(train_stats))]
    print (epochs)

    # # Create a plotly figure
    # fig = go.Figure()

    # # Add training loss line
    # fig.add_trace(go.Scatter(x=epochs, y=train_stats, mode='lines+markers', name='Train Loss', line=dict(color='blue')))

    # # Add validation loss line
    # fig.add_trace(go.Scatter(x=epochs, y=val_losses, mode='lines+markers', name='Val Loss', line=dict(color='red')))

    # # Customize the layout
    # fig.update_layout(
    #     title="Training and Validation Loss Over Epochs",
    #     xaxis_title="Epochs",
    #     yaxis_title="Loss",
    #     showlegend=True
    # )

    # # Show the plot
    # fig.show()