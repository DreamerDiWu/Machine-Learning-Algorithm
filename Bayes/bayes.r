demo_set <- function(){
	data <- list(
		c(1, 'S', -1),
		c(1, 'M', -1),
		c(1, 'M',  1),
		c(1, 'S',  1),
		c(1, 'S', -1),
		c(2, 'S', -1),
		c(2, 'M', -1),
		c(2, 'M',  1),
		c(2, 'L',  1) ,
		c(2, 'L',  1),
		c(3, 'L',  1),
		c(3, 'M',  1),
		c(3, 'M',  1),
		c(3, 'L',  1),
		c(3, 'L', -1)
		)
	return(data)
}

label_split <- function(dataSet){
	# extract labels from dataSet( accept 'list' type )
	# return labels

	# check input
	if (class(dataSet) != "list"){
		print("TypeError: parameter only accept class 'list'!")
		return()
	}
	len <- length(dataSet)
	label_ind = length(dataSet[[1]])
	# assign memory first
	labels <- rep(0, len)
	for ( i in 1:len ){
		labels[i] <- dataSet[[i]][label_ind]
	}
	return(labels)
}

# --------------#begin#----------------

demoSet <- demo_set()
labels <- label_split(demoSet)
demoSet
labels

# --------------#end#------------------

getPriorProb <- function(labels){
	tbl <- table(labels)
	prop.tbl <- prop.table(tbl)
	return(prop.tbl)
}

# --------------#begin#----------------

priorProb <- getPriorProb(labels)
priorProb

# --------------#end#------------------


getConditProb <- function(dataSet, labels, smoother=0){
	# notation: dataSet includes labels
	num_feats <- length(dataSet[[1]])-1
	label_tbl <- table(labels)
	label_name <- names(label_tbl)

	# a list to store unique value in each feature
	uniqVal <- list()
			
	# a list to store conditional probability
	conditProb <- list()

	# Count the example in different combination
	for (feat_ind in 1:num_feats){
		cur_uniqVal <- c()
		for (label_ind in 1:length(label_tbl)){
			for (examp_ind in 1:length(dataSet)){
				vec <- dataSet[[examp_ind]]
				value <- vec[feat_ind]
				# store unique value
				cur_uniqVal <- union(cur_uniqVal, value)
				# make a id like("2 'S' '1'") which means with condition y='1',
				# the prob of value of the 2nd feature that equals to 'S' 
				id <- paste(feat_ind, value, label_name[label_ind])

				# if the id first shows, make it zero
				if (is.null(conditProb[[id]])){conditProb[[id]] <- 0}
				
				
				# extract the true label of current example
				true_label <- as.character(vec[num_feats+1])

				if (true_label == label_name[label_ind]){
					conditProb[[id]] = conditProb[[id]] + 1
				}

			}
		}
		# store uniq value of current feature
		uniqVal[[feat_ind]] <- cur_uniqVal

	}

	for (i in 1:length(conditProb)){
		# parse the id to get number of uniq value
		id <- names(conditProb)[i]
		parsed_id <- unlist(strsplit(id, split=' '))
		feat_ind <- as.numeric(parsed_id[1])
		label <- parsed_id[3]
		num_uniqVal <- length(uniqVal[[feat_ind]])
		# according to the formula
		conditProb[[i]] <- ((conditProb[[i]] + smoother) / 
						(label_tbl[[label]] + num_uniqVal*smoother))

	}
	return(conditProb)
}

# --------------#begin#----------------

conditProb <- getConditProb(demoSet, labels, smoother=0)
conditProb
conditProb_smoothed <- getConditProb(demoSet, labels, smoother=1)
conditProb_smoothed

# --------------#end#------------------


bayes.train <- function(dataSet, smoother=0){
	# return a list with the priorProb and conditProb 
	labels <- label_split(dataSet)
	priorProb <- getPriorProb(labels)
	conditProb <- getConditProb(dataSet, labels, smoother)
	prob_lst <- list(priorProb=priorProb, conditProb=conditProb)
	return(prob_lst)
}

bayes.predict <- function(new_set, prob_lst){
	# parameter 'new_set' accept 'list' type only
	# return a list with predictions 
	if (class(new_set) != "list"){
		print("TypeError: parameter accept 'list' type only")
		return()
	}
	predictions <- list()
	num_feats <- length(new_set[[1]])
	priorProb <- prob_lst$priorProb
	conditProb <- prob_lst$conditProb

	for (i in 1:length(new_set)){
		cur_examp <- new_set[[i]]

		# initialize maximum prob and predition
		max_prob <- -Inf
		pred <- -999

		# calculation probs given each labels, store the maximum prob and correspond label 
		for (j in 1:length(priorProb)){
			# small number multiplies many time may vanish at zero
			# to prevent vanishing, use log to make summation
			baseProb <- log(priorProb[[j]])
			for (feat_ind in 1:num_feats){
				# make id to find corresponding prob in conditProb
				cur_id <- paste(feat_ind, cur_examp[feat_ind], names(priorProb)[j])
				cur_feat_conditProb <- conditProb[[cur_id]]

				# calculate probability according to the formula
				baseProb <- baseProb + log(cur_feat_conditProb)
			}
			print(cur_id)
			print(baseProb)
			# after calculate the prob given current label, compare with maximum prob
			if (baseProb > max_prob){
				max_prob <- baseProb
				pred <- names(priorProb[j])
			}
		}
		# store predition of current example to 'predictions'
		predictions[[i]] <- pred
	}

	return(predictions)
}

# --------------#begin#----------------

dataSet <- demo_set()
prob_lst <- bayes.train(dataSet, smoother=0)
new_set <- list(c(2, 'S'),c(3, 'M'),c(1, 'L'))
predictions <- bayes.predict(new_set, prob_lst)
predictions

# --------------#end#------------------
