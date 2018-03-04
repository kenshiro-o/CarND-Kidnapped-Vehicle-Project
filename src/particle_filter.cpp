/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 10;
	particles.resize(num_particles);
	weights.resize(num_particles);

	// Now initiliase all particles to values sampled from the distributions
	// given as parameters above
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	double initial_weight = 1.0;
	for(unsigned int i = 0; i < num_particles; ++i){
		Particle p = Particle();
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = initial_weight;	

		weights[i]	= initial_weight;
	}	
	is_initialized = true;	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	default_random_engine gen;

	for(unsigned int i = 0; i < particles.size(); ++i){
		Particle p = particles[i];
		double new_x;
		double new_y;
		double new_theta;
		if(fabs(yaw_rate) <= 0.00001){
			new_x = p.x + (velocity * delta_t) * cos(yaw_rate);
			new_y = p.y + (velocity * delta_t) * sin(yaw_rate);
			new_theta = yaw_rate;

		}else{
			double theta_dt = yaw_rate * delta_t;
			new_theta = p.theta + theta_dt;
			double v_over_yaw = velocity / yaw_rate;
			new_x = p.x + v_over_yaw * (sin(new_theta) - sin(p.theta));
			new_y = p.y + v_over_yaw * (cos(p.theta) - cos(new_theta));
		}

		normal_distribution<double> dist_x(new_x, std_x);
		normal_distribution<double> dist_y(new_y, std_y);
		normal_distribution<double> dist_theta(new_theta, std_theta);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);

		// Re-assign to the particles vector as structs are obtained by copy
		particles[i] = p;

		// cout << "Particle prediction [x=" <<  p.x << ", y=" << p.y << ", theta=" << p.theta << "]" << endl;
	}

	
	// cout << std_pos[0] << ", " << std_pos[1] << ", " << std_pos[2] << endl;

}

void ParticleFilter::dataAssociation(Particle &p, double std_landmark[], std::vector<LandmarkObs> &predicted, const std::vector<LandmarkObs> &observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// For each predicted landmark, find the map landmark that minimises the distance
	numeric_limits<double> nl;

	double std_x = std_landmark[0];
	double std_y = std_landmark[1]; 

	// cout << "STANDARD DEVIATION " << std_x << ", " << std_y << endl;
	// cout << "OBS SIZE " << predicted.size() << endl;

	std::vector<int> associations(predicted.size());
	std::vector<double> sense_x(predicted.size());
	std::vector<double> sense_y(predicted.size());
	p.weight = 1.0;
	// cout << "[dataAssociation] BEFORE LOOP" << endl;

	for(unsigned int i = 0; i < predicted.size(); ++i){
		LandmarkObs pred = predicted[i];
		double min_distance = nl.max();
		// cout << "MAX DOUBLE = " << min_distance << endl;
		LandmarkObs closest_landark;

		for(unsigned int j = 0; j < observations.size(); ++j){	
			LandmarkObs obs = observations[j];
			// cout << "OBS (" << obs.x << "," << obs.y << ", id=" << obs.id << ")" << endl;
			// cout << "POS (" << pred.x << ", " << pred.y << ", id=" << pred.id << ")" << endl;
			double dist = pred.distance(obs);
			if(dist < min_distance){				
				min_distance = dist;
				closest_landark = obs;				
			}
		}

		cout << "Found closest landmark [p-id=" << p.id <<", id=" << closest_landark.id << ", dist=" << min_distance << "]" << endl;
		cout << "pred coord (" << pred.x << "," << pred.y << ") - l coord (" << closest_landark.x << ", " << closest_landark.y << ")" << endl;

		associations[i] = closest_landark.id;
		sense_x[i] = pred.x;
		sense_y[i] = pred.y;		

		double w = multivariate_gaussian(pred.x, pred.y, 
			closest_landark.x, closest_landark.y, std_x, std_y);
		
		cout << "Computed multivariate gaussian [p-id=" << p.id << ", v=" << w  << "]" << endl;
		p.weight *= w;		
	}

	// cout << "[dataAssociation] AFTER LOOP" << endl;
	SetAssociations(p, associations, sense_x, sense_y);
} 

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Take the map landmarks and convert them into a an observation
	vector<LandmarkObs> map_real_landmarks(map_landmarks.landmark_list.size());
	for(unsigned int i = 0; i < map_real_landmarks.size(); ++i){
		auto m = map_landmarks.landmark_list[i];
		LandmarkObs l = LandmarkObs();
		l.id = m.id_i;
		l.x = m.x_f;
		l.y = m.y_f;
		map_real_landmarks[i] = l;
	}

	// cout << "MAP LANDMARK SIZE = " << map_real_landmarks.size() << endl;
	// cout << "OBSERVATIONS SIZE = " << observations.size() << endl;
		
	for(unsigned int i = 0; i < particles.size(); ++i){
		Particle p = particles[i];
		// cout << "P POS (" << p.x << ", " << p.y << ", id=" << p.id << ")" << endl;

		vector<LandmarkObs> particle_obs(observations.size());		
		for(unsigned int j = 0; j < observations.size(); ++j){
			// Convert the observation from car's coordinate system to map one, with respect to particle
			LandmarkObs obs = observations[j];
			LandmarkObs map_obs = obs.to_map_coordinates(p.x, p.y, p.theta);
			// cout << "CAR OBS (" << obs.x << ", " << obs.y << ", id=" << obs.id << ")" << endl;
			// cout << "MAP OBS (" << map_obs.x << ", " << map_obs.y << ", id=" << map_obs.id << ")" << endl;

			particle_obs[j] = 	map_obs;		
		}
		dataAssociation(p, std_landmark, particle_obs, map_real_landmarks); 		
		// cout << "FINISHED ONE DATA ASSOCIATION" << endl;
		
		particles[i] = p;
	}

	// cout << "CUPDATED WEIGHTS FOR ALL PARTICLES" << endl;

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// First calculate the sum of all weights
	double w_sum = 0.0;
	for(unsigned int i = 0; i < particles.size(); ++i){
		w_sum += particles[i].weight;
	}	

	// Normalise our weights so that their sum cumulates to 1	
	for(unsigned int i = 0; i < particles.size(); ++i){
		Particle p = particles[i];
		p.weight = p.weight / w_sum;
		// particles[i] = p;

		weights[i] = p.weight;
		cout << "Weight update: " << weights[i] << endl;
	}

	

	default_random_engine gen;
	discrete_distribution<unsigned int> dd(weights.begin(), weights.end());
	
	vector<Particle> survived_particles(particles.size());
	for(unsigned int i = 0; i < survived_particles.size(); ++i){
		int idx = dd(gen);
		cout << "INDEX SURVIVOR: " << idx << endl;
		survived_particles[i] = particles[idx];
	}

	// Our sampled (or survived particles) now become our new particles
	particles = survived_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
	
	// cout << "ABOUT TO SET ASSOCIATIONS " << endl;
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	// cout << "FINISHED SETTING ASSOCIATIONS " << endl;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // geDOUBLE_MAXt rid of the trailing space
    return s;
}
